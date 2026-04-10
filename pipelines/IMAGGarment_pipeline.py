from typing import Any, Callable, Dict, List, Optional, Union
import inspect
import os
import sys

import torch
from PIL import Image

from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import is_accelerate_available, logging
from diffusers.pipelines.controlnet.pipeline_controlnet import *
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import *
from diffusers.loaders import LoraLoaderMixin

from adapter.attention_processor import LogoRefSAttnProcessor2_0, IPAttnProcessor2_0
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class IMAGGarment(StableDiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae,
        reference_unet,
        unet,
        tokenizer,
        text_encoder,
        image_encoder,
        texture_ckpt,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
    ):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor)

        self.register_modules(
            vae=vae,
            reference_unet=reference_unet,
            unet=unet,
            scheduler=scheduler,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            image_encoder=image_encoder,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.clip_image_processor = CLIPImageProcessor()
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.ref_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False,
        )
        self.cond_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_convert_rgb=True,
            do_normalize=False,
        )

        # texture adapter
        self.texture_ckpt = texture_ckpt
        self.num_tokens = 4
        self.image_proj_model = self.init_proj()
        self.load_texture_adapter()

    def init_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    def load_texture_adapter(self):
        """
        Compatible with these checkpoint formats:
        1) training output:
           {
               "image_proj": ...,
               "texture_adapter": ...
           }

        2) old color adapter:
           {
               "image_proj": ...,
               "color_adapter": ...
           }

        3) old ip-adapter style:
           {
               "image_proj": ...,
               "ip_adapter": ...
           }

        4) safetensors with keys like:
           image_proj.xxx
           texture_adapter.xxx
           color_adapter.xxx
           ip_adapter.xxx
        """
        if self.texture_ckpt is None or self.texture_ckpt == "":
            raise ValueError("self.texture_ckpt is empty. Please provide a valid adapter checkpoint path.")

        ext = os.path.splitext(self.texture_ckpt)[-1].lower()

        if ext == ".safetensors":
            from safetensors import safe_open

            state_dict = {
                "image_proj": {},
                "texture_adapter": {},
                "color_adapter": {},
                "ip_adapter": {},
            }

            with safe_open(self.texture_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("texture_adapter."):
                        state_dict["texture_adapter"][key.replace("texture_adapter.", "")] = f.get_tensor(key)
                    elif key.startswith("color_adapter."):
                        state_dict["color_adapter"][key.replace("color_adapter.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.texture_ckpt, map_location="cpu")

        if "image_proj" not in state_dict or len(state_dict["image_proj"]) == 0:
            raise KeyError(
                f"'image_proj' not found in checkpoint: {self.texture_ckpt}. "
                f"Available keys: {list(state_dict.keys())}"
            )

        try:
            self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=False)
            print(f"[load_texture_adapter] loaded image_proj from: {self.texture_ckpt}")
        except RuntimeError as e:
            print("[load_texture_adapter] WARNING: image_proj shape mismatch, skipped loading image_proj.")
            print(f"[load_texture_adapter] details: {e}")
            print(
                "[load_texture_adapter] This usually means the inference image_encoder is different from the training image_encoder."
            )

        if "texture_adapter" in state_dict and len(state_dict["texture_adapter"]) > 0:
            adapter_sd = state_dict["texture_adapter"]
            adapter_name = "texture_adapter"
        elif "color_adapter" in state_dict and len(state_dict["color_adapter"]) > 0:
            adapter_sd = state_dict["color_adapter"]
            adapter_name = "color_adapter"
        elif "ip_adapter" in state_dict and len(state_dict["ip_adapter"]) > 0:
            adapter_sd = state_dict["ip_adapter"]
            adapter_name = "ip_adapter"
        else:
            raise KeyError(
                f"No adapter weights found in checkpoint: {self.texture_ckpt}. "
                f"Available keys: {list(state_dict.keys())}"
            )

        ip_layers = torch.nn.ModuleList(self.unet.attn_processors.values())
        missing, unexpected = ip_layers.load_state_dict(adapter_sd, strict=False)

        print(f"[load_texture_adapter] loaded adapter branch: {adapter_name}")
        if len(missing) > 0:
            print(f"[load_texture_adapter] missing keys: {len(missing)}")
        if len(unexpected) > 0:
            print(f"[load_texture_adapter] unexpected keys: {len(unexpected)}")

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(self.text_encoder, lora_scale)
            else:
                scale_lora_layers(self.text_encoder, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1: -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            if clip_skip is None:
                prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
                prompt_embeds = prompt_embeds[0]
            else:
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
                )
                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

        if self.text_encoder is not None:
            prompt_embeds_dtype = self.text_encoder.dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if isinstance(self, LoraLoaderMixin) and USE_PEFT_BACKEND:
            unscale_lora_layers(self.text_encoder, lora_scale)

        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        width,
        height,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_condition(
        self,
        cond_image,
        width,
        height,
        device,
        dtype,
        do_classififer_free_guidance=False,
    ):
        image = self.cond_image_processor.preprocess(cond_image, height=height, width=width).to(dtype=torch.float32)
        image = image.to(device=device, dtype=dtype)

        if do_classififer_free_guidance:
            image = torch.cat([image] * 2)

        return image

    @torch.inference_mode()
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_image_embeds = self.image_encoder(
                clip_image.to(self.device, dtype=torch.float16)
            ).image_embeds
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)

        image_prompt_embeds = self.image_proj_model(clip_image_embeds)
        uncond_image_prompt_embeds = self.image_proj_model(torch.zeros_like(clip_image_embeds))
        return image_prompt_embeds, uncond_image_prompt_embeds

    def set_scale(self, sketch_scale):
        for attn_processor in self.unet.attn_processors.values():
            if isinstance(attn_processor, LogoRefSAttnProcessor2_0):
                attn_processor.scale = sketch_scale

    def set_ipa_scale(self, ipa_scale):
        for attn_processor in self.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor2_0):
                attn_processor.scale = ipa_scale

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        null_prompt,
        negative_prompt,
        ref_image,
        width,
        height,
        num_inference_steps,
        guidance_scale,
        texture_clip_image=None,
        texture_embeds=None,
        ref_clip_image=None,
        num_images_per_prompt=1,
        sketch_scale=1.0,
        ipa_scale=1.0,
        num_samples=1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        clip_skip: Optional[int] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        self.set_scale(sketch_scale)
        self.set_ipa_scale(ipa_scale)
        self._guidance_scale = guidance_scale

        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        device = self._execution_device
        self._cross_attention_kwargs = cross_attention_kwargs
        self._clip_skip = clip_skip
        do_classifier_free_guidance = guidance_scale > 1.0

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        batch_size = 1

        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self._clip_skip,
        )

        image_prompt_embeds = None
        uncond_image_prompt_embeds = None

        if texture_clip_image is not None or texture_embeds is not None:
            image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
                pil_image=texture_clip_image,
                clip_image_embeds=texture_embeds,
            )

            bs_embed, seq_len, _ = image_prompt_embeds.shape
            image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
            image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

            uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
            uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

            prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)

            if do_classifier_free_guidance:
                negative_prompt_embeds = torch.cat(
                    [negative_prompt_embeds, uncond_image_prompt_embeds], dim=1
                )

        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            width,
            height,
            prompt_embeds.dtype,
            device,
            generator,
        )

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        ref_image_tensor = ref_image.to(dtype=self.vae.dtype, device=self.vae.device)
        ref_image_latents = self.vae.encode(ref_image_tensor).latent_dist.mean
        ref_image_latents = ref_image_latents * 0.18215

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if i == 0:
                    _ = self.reference_unet(
                        ref_image_latents,
                        torch.zeros_like(t),
                        encoder_hidden_states=None,
                        return_dict=False,
                    )

                    sa_hidden_states = {}
                    for name in self.reference_unet.attn_processors.keys():
                        if "attn1" in name:
                            sa_hidden_states[name] = self.reference_unet.attn_processors[name].cache["hidden_states"]

                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                timestep_cond = None
                if self.unet.config.time_cond_proj_dim is not None:
                    guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(
                        batch_size * num_images_per_prompt
                    )
                    timestep_cond = self.get_guidance_scale_embedding(
                        guidance_scale_tensor,
                        embedding_dim=self.unet.config.time_cond_proj_dim,
                    ).to(device=device, dtype=latents.dtype)

                noise_pred = self.unet(
                    latent_model_input[0].unsqueeze(0),
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs={"sa_hidden_states": sa_hidden_states},
                    timestep_cond=timestep_cond,
                    added_cond_kwargs=None,
                    return_dict=False,
                )[0]

                if do_classifier_free_guidance:
                    unc_noise_pred = self.unet(
                        latent_model_input[1].unsqueeze(0),
                        t,
                        encoder_hidden_states=negative_prompt_embeds,
                        timestep_cond=timestep_cond,
                        added_cond_kwargs=None,
                        return_dict=False,
                    )[0]

                    noise_pred_uncond, noise_pred_text = unc_noise_pred, noise_pred
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        latents = latents / self.vae.config.scaling_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = self.numpy_to_pil(image)
        return image
