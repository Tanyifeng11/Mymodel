from typing import Any, Callable, Dict, List, Optional, Union
import inspect
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
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
from models.bf_texture_module import BFTextureConditioner
from texture_preprocess import preprocess_texture_image
from checkpoint_utils import extract_texture_metadata, infer_texture_num_tokens, infer_clip_embed_dim

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
        spatial_texture_encoder=None,
        spatial_sketch_encoder=None,
        spatial_fusion=None,
        spatial_injection=None,
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
            spatial_texture_encoder=spatial_texture_encoder,
            spatial_sketch_encoder=spatial_sketch_encoder,
            spatial_fusion=spatial_fusion,
            spatial_injection=spatial_injection,
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
        self.num_tokens = 16
        self.texture_condition_mode = "patch_resampled"
        self.bf_texture_conditioner = None
        self.bf_clip_embeddings_dim = None
        self.image_proj_model = self.init_proj()
        self.texture_meta = {}
        self.effective_texture_num_tokens = self.num_tokens
        self.default_texture_condition_mode = "token"
        self.load_texture_adapter()

    def init_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.float16)
        return image_proj_model

    def _align_clip_embeds_dim(self, clip_image_embeds: torch.Tensor, expected_dim: int) -> torch.Tensor:
        """
        Align clip_image_embeds last dimension to expected_dim.
        - If actual > expected: truncate
        - If actual < expected: zero-pad
        """
        actual_dim = clip_image_embeds.shape[-1]

        if actual_dim == expected_dim:
            return clip_image_embeds

        print(
            f"[align_clip_embeds_dim] WARNING: clip_image_embeds dim mismatch: "
            f"actual={actual_dim}, expected={expected_dim}"
        )

        if actual_dim > expected_dim:
            print(f"[align_clip_embeds_dim] Truncating embedding dim from {actual_dim} -> {expected_dim}")
            return clip_image_embeds[..., :expected_dim]

        pad = expected_dim - actual_dim
        print(f"[align_clip_embeds_dim] Padding embedding dim from {actual_dim} -> {expected_dim}")
        return F.pad(clip_image_embeds, (0, pad))

    def load_texture_adapter(self):
        """
        Compatible with these checkpoint formats:
        1) training output:
           {
               "image_proj": ...,
               "texture_adapter": ...
           }

        2) BF texture output:
           {
               "bf_texture_conditioner": ...,
               "texture_adapter": ...
           }

        3) old color adapter:
           {
               "image_proj": ...,
               "color_adapter": ...
           }

        4) old ip-adapter style:
           {
               "image_proj": ...,
               "ip_adapter": ...
           }

        5) safetensors with keys like:
           image_proj.xxx
           bf_texture_conditioner.xxx
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
                "bf_texture_conditioner": {},
                "texture_adapter": {},
                "color_adapter": {},
                "ip_adapter": {},
            }

            with safe_open(self.texture_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("bf_texture_conditioner."):
                        state_dict["bf_texture_conditioner"][key.replace("bf_texture_conditioner.", "")] = f.get_tensor(key)
                    elif key.startswith("texture_adapter."):
                        state_dict["texture_adapter"][key.replace("texture_adapter.", "")] = f.get_tensor(key)
                    elif key.startswith("color_adapter."):
                        state_dict["color_adapter"][key.replace("color_adapter.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            # GAM.pt 之前已经验证过不适合 weights_only=True
            # 这里 texture adapter 也先保持 False，避免再遇到反序列化限制问题
            state_dict = torch.load(self.texture_ckpt, map_location="cpu")

        self.texture_meta = extract_texture_metadata(state_dict)
        if self.texture_meta:
            print(f"[load_texture_adapter] metadata: {self.texture_meta}")

        if "image_proj" in state_dict and len(state_dict["image_proj"]) > 0:
            self.texture_condition_mode = "patch_resampled"
            try:
                self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=False)
                self.effective_texture_num_tokens = int(self.texture_meta.get("texture_num_tokens", self.num_tokens))
                print(f"[load_texture_adapter] loaded image_proj from: {self.texture_ckpt}")
            except RuntimeError as e:
                print("[load_texture_adapter] WARNING: image_proj shape mismatch, skipped loading image_proj.")
                print(f"[load_texture_adapter] details: {e}")
                print(
                    "[load_texture_adapter] This usually means the inference image_encoder is different from the training image_encoder."
                )

        elif "bf_texture_conditioner" in state_dict and len(state_dict["bf_texture_conditioner"]) > 0:
            self.texture_condition_mode = "patch_resampled"
            bf_sd = state_dict["bf_texture_conditioner"]
            c1 = bf_sd["stage1.0.weight"].shape[0]
            c2 = bf_sd["stage2.0.weight"].shape[0]
            c3 = bf_sd["stage3.0.weight"].shape[0]
            c4 = bf_sd["stage4.0.weight"].shape[0]
            num_tokens = infer_texture_num_tokens(state_dict, default=self.num_tokens)
            clip_embed_dim = infer_clip_embed_dim(state_dict, fallback=self.image_encoder.config.hidden_size)

            self.bf_texture_conditioner = BFTextureConditioner(
                clip_embeddings_dim=clip_embed_dim,
                cross_attention_dim=self.unet.config.cross_attention_dim,
                num_tokens=num_tokens,
                stage_channels=(c1, c2, c3, c4),
                texture_mode="patch_resampled",
            ).to(self.device, dtype=torch.float16)

            missing, unexpected = self.bf_texture_conditioner.load_state_dict(bf_sd, strict=False)
            self.effective_texture_num_tokens = num_tokens
            print(
                f"[load_texture_adapter] loaded bf_texture_conditioner from: {self.texture_ckpt} "
                f"(num_tokens={num_tokens}, clip_embed_dim={clip_embed_dim}, stage_channels={(c1, c2, c3, c4)}, missing={len(missing)}, unexpected={len(unexpected)})"
            )

        else:
            raise KeyError(
                f"No supported texture conditioner found in checkpoint: {self.texture_ckpt}. "
                f"Available keys: {list(state_dict.keys())}"
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
    def get_image_embeds(self, pil_image=None, clip_image_embeds=None, width=None, height=None, texture_mode="patch_resampled"):
        clip_patch_tokens = None
        if pil_image is not None:
            if isinstance(pil_image, Image.Image):
                pil_image = [pil_image]
            clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
            clip_outputs = self.image_encoder(
                clip_image.to(self.device, dtype=torch.float16),
                output_hidden_states=True,
            )
            clip_image_embeds = clip_outputs.image_embeds
            clip_patch_tokens = clip_outputs.hidden_states[-1][:, 1:, :]
        else:
            clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)

        use_bf = self.bf_texture_conditioner is not None
        if use_bf:
            if pil_image is None:
                raise ValueError("BF texture conditioning requires PIL texture image input.")
            if width is None or height is None:
                raise ValueError("width/height must be provided for BF texture conditioning.")

            texture_tensor = self.cond_image_processor.preprocess(
                pil_image, height=height, width=width
            ).to(self.device, dtype=torch.float16)
            texture_tensor = texture_tensor * 2.0 - 1.0

            image_prompt_embeds, _ = self.bf_texture_conditioner(
                clip_image_embeds=clip_image_embeds,
                texture_images=texture_tensor,
                clip_vision_tokens=clip_patch_tokens,
                texture_mode=texture_mode,
            )

            zero_clip = torch.zeros_like(clip_image_embeds)
            zero_patch = torch.zeros_like(clip_patch_tokens) if clip_patch_tokens is not None else None
            zero_texture = torch.zeros_like(texture_tensor)
            uncond_image_prompt_embeds, _ = self.bf_texture_conditioner(
                clip_image_embeds=zero_clip,
                texture_images=zero_texture,
                clip_vision_tokens=zero_patch,
                texture_mode=texture_mode,
            )
        else:
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
        texture_mode="patch_resampled",
        texture_condition_mode="spatial",
        fusion_type="minimal",
        texture_preprocess_mode="crop_tile",
        texture_num_tokens=16,
        texture_scale=1.0,
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
            force_override = kwargs.get("force_texture_num_tokens_override", False)
            use_token = texture_condition_mode in ("token", "hybrid")
            use_spatial = texture_condition_mode in ("spatial", "hybrid")
            ckpt_tokens = self.effective_texture_num_tokens
            if texture_num_tokens != ckpt_tokens:
                if force_override:
                    print(f"[IMAGGarment][WARNING] forcing texture_num_tokens from checkpoint {ckpt_tokens} to CLI {texture_num_tokens}.")
                    ckpt_tokens = texture_num_tokens
                else:
                    print(f"[IMAGGarment][WARNING] CLI texture_num_tokens={texture_num_tokens} but checkpoint uses {ckpt_tokens}. using checkpoint value.")
            texture_num_tokens = ckpt_tokens
            if use_token:
                for attn_processor in self.unet.attn_processors.values():
                    if isinstance(attn_processor, IPAttnProcessor2_0):
                        attn_processor.num_tokens = texture_num_tokens
            print(f"[IMAGGarment] checkpoint format: {self.texture_meta.get('checkpoint_format', 'texture_adapter')}")
            print(f"[IMAGGarment] texture mode: {texture_mode}")
            print(f"[IMAGGarment] texture condition mode: {texture_condition_mode}")
            print(f"[IMAGGarment] texture ckpt path: {self.texture_ckpt}")
            if use_token:
                print(f"[IMAGGarment] texture token count: {texture_num_tokens}")
                image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(
                    pil_image=texture_clip_image,
                    clip_image_embeds=texture_embeds,
                    width=width,
                    height=height,
                    texture_mode=texture_mode,
                )
                image_prompt_embeds = image_prompt_embeds * texture_scale
                uncond_image_prompt_embeds = uncond_image_prompt_embeds * texture_scale

                bs_embed, seq_len, _ = image_prompt_embeds.shape
                image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
                image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

                uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
                uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)

                prompt_embeds = torch.cat([prompt_embeds, image_prompt_embeds], dim=1)
                print(f"[IMAGGarment] final encoder_hidden_states shape: {tuple(prompt_embeds.shape)}")

                if do_classifier_free_guidance:
                    negative_prompt_embeds = torch.cat(
                        [negative_prompt_embeds, uncond_image_prompt_embeds], dim=1
                    )

            if use_spatial and all(
                m is not None
                for m in [self.spatial_texture_encoder, self.spatial_sketch_encoder, self.spatial_fusion, self.spatial_injection]
            ):
                tex_img = texture_clip_image if isinstance(texture_clip_image, Image.Image) else texture_clip_image[0]
                tex_tensor = preprocess_texture_image(
                    tex_img.convert("RGB"),
                    width=width,
                    height=height,
                    mode=texture_preprocess_mode,
                ).unsqueeze(0).to(device=device, dtype=torch.float16)
                sketch_tensor = ref_image.to(device=device, dtype=tex_tensor.dtype)
                if sketch_tensor.shape[-2:] != tex_tensor.shape[-2:]:
                    sketch_tensor = F.interpolate(sketch_tensor, size=tex_tensor.shape[-2:], mode="bilinear", align_corners=False)
                sketch_feats = self.spatial_sketch_encoder(sketch_tensor)
                texture_feats = self.spatial_texture_encoder(tex_tensor)
                if hasattr(self.spatial_fusion, "set_fusion_type"):
                    self.spatial_fusion.set_fusion_type(fusion_type)
                spatial_feats = self.spatial_fusion(sketch_feats, texture_feats)
                self.spatial_injection.set_alphas([
                    kwargs.get("alpha1", 1.0),
                    kwargs.get("alpha2", 1.0),
                    kwargs.get("alpha3", 0.7),
                    kwargs.get("alpha4", 0.5),
                ])
                self.spatial_injection.set_features(spatial_feats)
                self.spatial_injection.enable()

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
                    if texture_condition_mode in ("spatial", "hybrid") and self.spatial_injection is not None:
                        self.spatial_injection.clear_features()
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
        if self.spatial_injection is not None:
            self.spatial_injection.clear_features()
            self.spatial_injection.disable()
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = self.numpy_to_pil(image)
        return image
