import argparse
import itertools
import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from accelerate import Accelerator
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPImageProcessor

from adapter.attention_processor import LogoCacheSAttnProcessor2_0, CAttnProcessor2_0, LogoRefSAttnProcessor2_0, LogoCacheCAttnProcessor2_0, IPAttnProcessor2_0
from models.bf_texture_module import BFTextureConditioner


class JointTextureDataset(Dataset):
    def __init__(self, json_path, tokenizer, image_root):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.image_root = image_root
        self.vae_tf = transforms.Compose([
            transforms.Resize((640, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
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
        input_ids = self.tokenizer(caption, padding="max_length", truncation=True, max_length=self.tokenizer.model_max_length, return_tensors="pt").input_ids[0]
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


def set_unet_trainable(unet):
    for p in unet.parameters():
        p.requires_grad = False
    for n, p in unet.named_parameters():
        if "mid_block" in n and ("attn" in n or "transformer" in n):
            p.requires_grad = True
        if "up_blocks.2" in n or "up_blocks.3" in n:
            if "attn" in n or "transformer" in n:
                p.requires_grad = True
    for proc in unet.attn_processors.values():
        for p in proc.parameters():
            p.requires_grad = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_model_name_or_path", required=True)
    ap.add_argument("--pretrained_vae_model_path", required=True)
    ap.add_argument("--image_encoder_path", required=True)
    ap.add_argument("--dataset_json_path", required=True)
    ap.add_argument("--data_root_path", required=True)
    ap.add_argument("--texture_adapter_ckpt", required=True)
    ap.add_argument("--output_dir", default="joint_texture_output")
    ap.add_argument("--train_batch_size", type=int, default=4)
    ap.add_argument("--max_train_steps", type=int, default=20000)
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--num_warmup_steps", type=int, default=500)
    ap.add_argument("--bf_num_tokens", type=int, default=16)
    args = ap.parse_args()

    accelerator = Accelerator()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    ref_unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    vae = AutoencoderKL.from_pretrained(args.pretrained_vae_model_path)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)

    attn_procs = {}
    for name in unet.attn_processors.keys():
        cad = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hs = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            hs = list(reversed(unet.config.block_out_channels))[int(name[len("up_blocks.")])]
        else:
            hs = unet.config.block_out_channels[int(name[len("down_blocks.")])]
        attn_procs[name] = LogoRefSAttnProcessor2_0(name, hs) if cad is None else IPAttnProcessor2_0(hs, cad, num_tokens=args.bf_num_tokens)
    unet.set_attn_processor(attn_procs)

    attn_procs2 = {}
    for name in ref_unet.attn_processors.keys():
        cad = None if name.endswith("attn1.processor") else ref_unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hs = ref_unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            hs = list(reversed(ref_unet.config.block_out_channels))[int(name[len("up_blocks.")])]
        else:
            hs = ref_unet.config.block_out_channels[int(name[len("down_blocks.")])]
        attn_procs2[name] = LogoCacheSAttnProcessor2_0(name, hs) if cad is None else LogoCacheCAttnProcessor2_0(name, hs, hs)
    ref_unet.set_attn_processor(attn_procs2)

    bf = BFTextureConditioner(clip_embeddings_dim=image_encoder.config.hidden_size, cross_attention_dim=unet.config.cross_attention_dim, num_tokens=args.bf_num_tokens)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())

    ckpt = torch.load(args.texture_adapter_ckpt, map_location="cpu")
    adapter_modules.load_state_dict(ckpt["texture_adapter"], strict=False)
    bf.load_state_dict(ckpt["bf_texture_conditioner"], strict=False)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)
    set_unet_trainable(unet)
    bf.requires_grad_(True)
    ref_unet.requires_grad_(True)

    params = itertools.chain((p for p in unet.parameters() if p.requires_grad), bf.parameters(), ref_unet.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate)
    scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=args.max_train_steps)

    ds = JointTextureDataset(args.dataset_json_path, tokenizer, args.data_root_path)
    dl = DataLoader(ds, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)

    unet, ref_unet, bf, optimizer, dl, scheduler = accelerator.prepare(unet, ref_unet, bf, optimizer, dl, scheduler)
    text_encoder.to(accelerator.device)
    vae.to(accelerator.device)
    image_encoder.to(accelerator.device)

    ns = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, prediction_type="epsilon")
    step = 0
    while step < args.max_train_steps:
        for batch in dl:
            with torch.no_grad():
                latents = vae.encode(batch["vae_cloth"]).latent_dist.sample() * 0.18215
                ref_latents = vae.encode(batch["vae_sketch"]).latent_dist.sample() * 0.18215
                text_h = text_encoder(batch["input_ids"])[0]
                clip_out = image_encoder(batch["clip_texture"], output_hidden_states=True)

            tex_tokens, _ = bf(clip_image_embeds=clip_out.image_embeds, texture_images=batch["texture_image"], clip_vision_tokens=clip_out.hidden_states[-1][:, 1:, :], texture_mode="patch_resampled")
            enc_h = torch.cat([text_h, tex_tokens], dim=1)

            noise = torch.randn_like(latents)
            t = torch.randint(0, ns.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
            noisy_latents = ns.add_noise(latents, noise, t)

            _ = ref_unet(ref_latents, torch.zeros_like(t), None, return_dict=False)
            sa = {n: ref_unet.attn_processors[n].cache["hidden_states"] for n in ref_unet.attn_processors.keys() if "attn1" in n}
            pred = unet(noisy_latents, t, encoder_hidden_states=enc_h, cross_attention_kwargs={"sa_hidden_states": sa}).sample
            loss = F.mse_loss(pred.float(), noise.float(), reduction="mean")

            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if accelerator.is_main_process and step % 100 == 0:
                print(f"step={step}, loss={loss.item():.6f}, encoder_hidden_states={tuple(enc_h.shape)}")
            if accelerator.is_main_process and step % 2000 == 0 and step > 0:
                torch.save({
                    "unet": accelerator.unwrap_model(unet).state_dict(),
                    "ref_unet": accelerator.unwrap_model(ref_unet).state_dict(),
                    "texture_adapter": accelerator.unwrap_model(torch.nn.ModuleList(unet.attn_processors.values())).state_dict(),
                    "bf_texture_conditioner": accelerator.unwrap_model(bf).state_dict(),
                }, os.path.join(args.output_dir, f"checkpoint-{step}.pt"))
            step += 1
            if step >= args.max_train_steps:
                break


if __name__ == "__main__":
    main()
