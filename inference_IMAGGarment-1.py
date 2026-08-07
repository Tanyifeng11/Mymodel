from pipelines.IMAGGarment_pipeline import IMAGGarment
import os
import json
import importlib
import importlib.util
import torch
import numpy as np

from PIL import Image
from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from torchvision import transforms
from transformers import CLIPImageProcessor
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from adapter.attention_processor import LogoCacheSAttnProcessor2_0, LogoRefSAttnProcessor2_0, LogoCacheCAttnProcessor2_0 , CAttnProcessor2_0,IPAttnProcessor2_0
from models.multiscale_texture_encoder import MultiScaleTextureEncoder
from models.palette_tokenizer import PaletteTokenMLP
from models.spatial_injection import SpatialInjectionAdapter
from models.attribute_text_texture_fuser import AttributeTextTextureFuser
import argparse
from garment_mask_utils import build_sketch_garment_mask

try:
    _repo_checkpoint_spec = importlib.util.find_spec("repo_utils.checkpoint_utils")
except ModuleNotFoundError:
    _repo_checkpoint_spec = None
_checkpoint_utils = importlib.import_module("repo_utils.checkpoint_utils") if _repo_checkpoint_spec is not None else importlib.import_module("checkpoint_utils")
load_checkpoint_file = _checkpoint_utils.load_checkpoint_file
detect_gam_checkpoint_format = _checkpoint_utils.detect_gam_checkpoint_format
infer_texture_num_tokens = _checkpoint_utils.infer_texture_num_tokens
extract_texture_metadata = _checkpoint_utils.extract_texture_metadata


def _get_layer_group(name: str) -> str:
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


def _set_balanced_gate_trace(pipe, enabled: bool):
    for name, proc in pipe.unet.attn_processors.items():
        if isinstance(proc, IPAttnProcessor2_0):
            proc.processor_name = name
            proc.balanced_gate_trace_enabled = bool(enabled)
            proc.balanced_gate_trace = []


def _save_balanced_gate_trace(pipe, trace_path: str, sample_id: str = ""):
    if not trace_path:
        return
    rows = []
    for name, proc in pipe.unet.attn_processors.items():
        if not isinstance(proc, IPAttnProcessor2_0):
            continue
        for row in getattr(proc, "balanced_gate_trace", []):
            out = {"sample_id": sample_id, **row}
            if not out.get("layer_name"):
                out["layer_name"] = name
            rows.append(out)
    os.makedirs(os.path.dirname(trace_path) or ".", exist_ok=True)
    with open(trace_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_image_encoder_flexible(image_encoder_path, device=None, dtype=None):
    try:
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path)
    except Exception:
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path, subfolder="models/image_encoder")
    if device is not None or dtype is not None:
        image_encoder = image_encoder.to(device=device, dtype=dtype)
    return image_encoder


def resolve_image_encoder_path(args):
    if args.image_encoder_path and args.image_encoder_path != "auto":
        return args.image_encoder_path
    if args.texture_ckpt:
        texture_state = load_checkpoint_file(args.texture_ckpt)
        texture_meta = extract_texture_metadata(texture_state)
        ckpt_path = texture_meta.get("image_encoder_path")
        if ckpt_path:
            return ckpt_path
    return "openai/clip-vit-large-patch14"


def resize_img(input_image, max_side=640, min_side=512, size=None,
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):
    w, h = input_image.size
    ratio = min_side / min(h, w)
    w, h = round(ratio * w), round(ratio * h)
    ratio = max_side / max(h, w)
    input_image = input_image.resize([round(ratio * w), round(ratio * h)], mode)
    w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
    h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    return input_image


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    max_w,max_h=0,0
    for img in imgs :
        max_w = max(max_w,img.size[0])
        max_h = max(max_h,img.size[1])
            
    w, h = max_w,max_h
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def load_gam_checkpoint(ckpt_path, unet, ref_unet, adapter_modules):
    state = load_checkpoint_file(ckpt_path)
    ckpt_format = detect_gam_checkpoint_format(state)
    print(f"[load_gam_checkpoint] detected format: {ckpt_format}")

    unet_loaded = ref_loaded = adapter_loaded = bf_loaded = False
    bf_state = None
    if ckpt_format == "legacy_module":
        model_sd = state["module"]
        ref_unet_dict = {}
        unet_dict = {}
        adapter_modules_dict = {}
        for k, v in model_sd.items():
            if k.startswith("ref_unet"):
                ref_unet_dict[k.replace("ref_unet.", "")] = v
            elif k.startswith("unet"):
                unet_dict[k.replace("unet.", "")] = v
            elif k.startswith("adapter_modules"):
                adapter_modules_dict[k.replace("adapter_modules.", "")] = v
        if unet_dict:
            unet.load_state_dict(unet_dict, strict=False)
            unet_loaded = True
        if ref_unet_dict:
            ref_unet.load_state_dict(ref_unet_dict, strict=False)
            ref_loaded = True
        if adapter_modules_dict:
            adapter_modules.load_state_dict(adapter_modules_dict, strict=False)
            adapter_loaded = True
        meta = {}
    elif ckpt_format in ("gam_texture_joint_v1", "gam_texture_joint_v2", "gam_texture_joint_v3") or all(
        k in state for k in ("unet", "ref_unet", "texture_adapter")
    ):
        if "unet" in state:
            unet.load_state_dict(state["unet"], strict=False)
            unet_loaded = True
        if "ref_unet" in state:
            ref_unet.load_state_dict(state["ref_unet"], strict=False)
            ref_loaded = True
        if "texture_adapter" in state:
            adapter_modules.load_state_dict(state["texture_adapter"], strict=False)
            adapter_loaded = True
        if "bf_texture_conditioner" in state:
            bf_state = state["bf_texture_conditioner"]
            bf_loaded = True
        meta = extract_texture_metadata(state)
    else:
        raise ValueError(f"Unsupported GAM checkpoint format: {ckpt_format}")

    print(f"[load_gam_checkpoint] unet_loaded={unet_loaded}, ref_unet_loaded={ref_loaded}, adapter_loaded={adapter_loaded}, bf_in_ckpt={bf_loaded}")
    if meta:
        print(f"[load_gam_checkpoint] metadata: {meta}")
    return {"format": ckpt_format, "meta": meta, "bf_state": bf_state, "state": state}


def prepare(args):
    if not args.texture_ckpt:
        args.texture_ckpt = args.GAM_model_ckpt
        print(f"[prepare] texture_ckpt is empty, using GAM_model_ckpt: {args.texture_ckpt}")

    gam_meta_for_paths = {}
    if args.base_model_path == "auto" or args.vae_model_path == "auto":
        try:
            gam_state_for_paths = load_checkpoint_file(args.GAM_model_ckpt)
            gam_meta_for_paths = extract_texture_metadata(gam_state_for_paths)
        except Exception as e:
            print(f"[WARNING] failed to read GAM metadata for base path auto-resolve: {e}")

    if args.base_model_path == "auto":
        args.base_model_path = gam_meta_for_paths.get(
            "pretrained_model_name_or_path",
            "stable-diffusion-v1-5/stable-diffusion-v1-5",
        )
    if args.vae_model_path == "auto":
        args.vae_model_path = gam_meta_for_paths.get(
            "pretrained_vae_model_path",
            "stabilityai/sd-vae-ft-mse",
        )

    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    resolved_image_encoder_path = resolve_image_encoder_path(args)
    print(f"[prepare] base model path: {args.base_model_path}")
    print(f"[prepare] vae model path: {args.vae_model_path}")
    print(f"[prepare] resolved image encoder path: {resolved_image_encoder_path}")
    print(f"[prepare] generation seed: {args.seed}")
    print(f"[prepare] layer_group_enabled = {bool(args.layer_group_enabled)}")
    print(f"[prepare] use_texture_gate = {bool(args.use_texture_gate)}")
    print(f"[prepare] use_palette_tokens = {bool(args.use_palette_tokens)}")
    print(f"[prepare] use_balanced_fusion_gate = {bool(args.use_balanced_fusion_gate)}")
    print(f"[prepare] use_tcpm_lite = {bool(args.use_tcpm_lite)}")
    print(f"[prepare] num_palette_tokens = {args.num_palette_tokens}")
    print(f"[prepare] gate_type = {args.gate_type}")
    print(f"[prepare] gate_init = {args.gate_init}")
    print(f"[prepare] gate_min = {args.gate_min}, gate_max = {args.gate_max}")
    
    # Keep inference base components aligned with training base components.
    vae = AutoencoderKL.from_pretrained(args.vae_model_path).to(dtype=torch.float16, device=args.device)
    tokenizer = CLIPTokenizer.from_pretrained(args.base_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.base_model_path, subfolder="text_encoder").to(
        dtype=torch.float16, device=args.device)
    unet = UNet2DConditionModel.from_pretrained(args.base_model_path, subfolder="unet").to(
        dtype=torch.float16,device=args.device)
    image_encoder = load_image_encoder_flexible(
        resolved_image_encoder_path,
        device=args.device,
        dtype=torch.float16,
    )

    # set attention processor
    attn_procs = {}
    st = unet.state_dict()
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
        if cross_attention_dim is None:
            attn_procs[name] = LogoRefSAttnProcessor2_0(name, hidden_size)
        else:
            if args.layer_group_enabled:
                layer_group = _get_layer_group(name)
                detail_text_scale = _get_detail_text_scale(name)
            else:
                layer_group = "all"
                detail_text_scale = 0.1
            attn_procs[name] = IPAttnProcessor2_0(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                num_tokens=args.texture_num_tokens,
                layer_group=layer_group,
                detail_text_scale=detail_text_scale,
                use_texture_gate=bool(args.use_texture_gate),
                gate_type=args.gate_type,
                gate_init=args.gate_init,
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
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    adapter_modules = adapter_modules.to(dtype=torch.float16, device=args.device)
    del st
    

    ref_unet = UNet2DConditionModel.from_pretrained(args.base_model_path, subfolder="unet").to(
        dtype=torch.float16,
        device=args.device)
    attn_procs2 = {}
    st = ref_unet.state_dict()
    for name in ref_unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else ref_unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = ref_unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(ref_unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = ref_unet.config.block_out_channels[block_id]
        # lora_rank = hidden_size // 2 # args.lora_rank
        if cross_attention_dim is None:
            attn_procs2[name] = LogoCacheSAttnProcessor2_0(name, hidden_size)
        else:
            attn_procs2[name] = LogoCacheCAttnProcessor2_0(name, hidden_size=hidden_size,
                                                 cross_attention_dim=cross_attention_dim)  # .to(accelerator.device)]
    ref_unet.set_attn_processor(attn_procs2)

    del st
    ref_unet.to(dtype=torch.float16,device=args.device)
    # weights load
    gam_info = load_gam_checkpoint(args.GAM_model_ckpt, unet, ref_unet, adapter_modules)
    gam_meta = gam_info.get("meta", {})
    ckpt_tokens = int(gam_meta.get("texture_num_tokens", args.texture_num_tokens))
    if ckpt_tokens != args.texture_num_tokens:
        if args.force_texture_num_tokens_override:
            print(f"[WARNING] force override texture_num_tokens: ckpt={ckpt_tokens}, cli={args.texture_num_tokens}")
        else:
            print(f"[WARNING] texture_num_tokens mismatch: ckpt={ckpt_tokens}, cli={args.texture_num_tokens}. using checkpoint value.")
            args.texture_num_tokens = ckpt_tokens

    ckpt_width = gam_meta.get("width", None)
    ckpt_height = gam_meta.get("height", None)
    if args.width is None:
        args.width = int(ckpt_width) if ckpt_width is not None else 512
    if args.height is None:
        args.height = int(ckpt_height) if ckpt_height is not None else 640
    if ckpt_width is not None and ckpt_height is not None:
        ckpt_width = int(ckpt_width)
        ckpt_height = int(ckpt_height)
        if args.width != ckpt_width or args.height != ckpt_height:
            print(
                f"[WARNING] inference resolution ({args.width}, {args.height}) "
                f"!= GAM checkpoint resolution ({ckpt_width}, {ckpt_height}). "
                f"建议保持一致以获得稳定结构控制。"
            )
    print(f"[prepare] effective inference resolution: width={args.width}, height={args.height}")

    for proc in unet.attn_processors.values():
        if isinstance(proc, IPAttnProcessor2_0):
            proc.num_tokens = args.texture_num_tokens
    print(f"[prepare] effective texture_num_tokens for IPAttnProcessor2_0: {args.texture_num_tokens}")

    gam_state = gam_info.get("state", {})
    aa_tcr_fuser = None
    aa_tcr_state = gam_state.get("aa_tcr_fuser", {})
    if args.use_aa_tcr_fuse:
        if not aa_tcr_state:
            raise RuntimeError(
                "use_aa_tcr_fuse=1 but GAM checkpoint has no aa_tcr_fuser state"
            )
        aa_tcr_fuser = AttributeTextTextureFuser(
            hidden_dim=unet.config.cross_attention_dim,
            num_heads=int(gam_meta.get("aa_tcr_num_heads", 4)),
            head_dim=gam_meta.get("aa_tcr_head_dim"),
            alpha_init=float(gam_meta.get("aa_tcr_alpha_init", 0.0)),
            max_alpha=gam_meta.get("aa_tcr_max_alpha"),
            empty_mask_fallback=bool(gam_meta.get("aa_tcr_empty_fallback", 1)),
        ).to(device=args.device, dtype=torch.float16)
        missing, unexpected = aa_tcr_fuser.load_state_dict(aa_tcr_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "AA-TCR checkpoint is incomplete: "
                f"missing={missing}, unexpected={unexpected}"
            )
        aa_tcr_fuser.eval()
        print("[prepare] restored AA-TCR Fuse from GAM checkpoint")

    spatial_texture_encoder = None
    spatial_injection = None
    if args.texture_condition_mode in ("spatial", "hybrid"):
        spatial_texture_encoder = MultiScaleTextureEncoder(stage_channels=(64, 128, 256, 256)).to(dtype=torch.float16, device=args.device)
        spatial_injection = SpatialInjectionAdapter(
            unet=unet,
            fusion_channels=(64, 128, 256, 256),
            target_channels=(unet.config.block_out_channels[0], unet.config.block_out_channels[1], unet.config.block_out_channels[2], unet.config.block_out_channels[-1]),
            alphas=(args.alpha1, args.alpha2, args.alpha3, args.alpha4),
        ).to(dtype=torch.float16, device=args.device)
        st = gam_info.get("state", {})
        spatial_loaded_flags = {
            "spatial_texture_encoder": False,
            "spatial_injection": False,
        }
        if "spatial_texture_encoder" in st:
            spatial_texture_encoder.load_state_dict(st["spatial_texture_encoder"], strict=False)
            spatial_loaded_flags["spatial_texture_encoder"] = True
        if "spatial_injection" in st:
            spatial_injection.load_state_dict(st["spatial_injection"], strict=False)
            spatial_loaded_flags["spatial_injection"] = True
        if not all(spatial_loaded_flags.values()):
            print(
                "[WARNING] Spatial branch weights are incomplete in GAM checkpoint: "
                f"{spatial_loaded_flags}. spatial/hybrid 可能无法正常发挥。"
            )

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    pipe = IMAGGarment(unet=unet, reference_unet=ref_unet, vae=vae, tokenizer=tokenizer,
                         text_encoder=text_encoder, image_encoder=image_encoder,
                         texture_ckpt=args.texture_ckpt,
                         spatial_texture_encoder=spatial_texture_encoder,
                         spatial_injection=spatial_injection,
                         use_texture_gate=bool(args.use_texture_gate),
                         use_tcpm_lite=bool(args.use_tcpm_lite),
                         tcpm_hidden_ratio=args.tcpm_hidden_ratio,
                         tcpm_residual_scale_init=args.tcpm_residual_scale_init,
                         aa_tcr_fuser=aa_tcr_fuser,
                         scheduler=noise_scheduler,
                         safety_checker=StableDiffusionSafetyChecker,
                         feature_extractor=CLIPImageProcessor)
    pipe.set_layer_group_enabled(bool(args.layer_group_enabled))
    pipe.tcpm_lite.to(dtype=torch.float16, device=args.device)

    # IMAGGarment will load args.texture_ckpt in __init__, which can overwrite
    # adapter/BF states already loaded from GAM checkpoint. Restore GAM states here.
    if "texture_adapter" in gam_state:
        adapter_sd = gam_state["texture_adapter"]
        checkpoint_has_gate = any(
            "texture_gate_delta" in key or ".gate" in key or key.startswith("gate")
            for key in adapter_sd.keys()
        )
        missing, unexpected = torch.nn.ModuleList(pipe.unet.attn_processors.values()).load_state_dict(
            adapter_sd, strict=False
        )
        gate_unexpected = [k for k in unexpected if "texture_gate_delta" in k or "gate" in k]
        gate_missing = [k for k in missing if "texture_gate_delta" in k or "gate" in k]
        palette_missing = [
            k for k in missing
            if "palette_branch_scale" in k or "to_k_palette" in k or "to_v_palette" in k
        ]
        if args.use_texture_gate:
            if not checkpoint_has_gate:
                print("[prepare] WARNING: use_texture_gate=1 but checkpoint has no texture gate parameters")
                raise RuntimeError("E2b requires texture gate parameters in GAM checkpoint")
            if gate_missing:
                print(f"[prepare] WARNING: texture gate parameters were not loaded: {gate_missing[:16]}")
                raise RuntimeError("E2b texture gate parameters are incomplete in GAM checkpoint")
            gate_keys = []
            for name, proc in pipe.unet.attn_processors.items():
                if hasattr(proc, "texture_gate_delta") and proc.texture_gate_delta is not None:
                    gate_keys.append(name)
            print(f"[prepare] Loaded texture gate parameters, count={len(gate_keys)}")
        elif gate_unexpected:
            print("[prepare] WARNING: checkpoint contains gate parameters but use_texture_gate=0")
        if palette_missing:
            print(f"[prepare] expected missing palette keys: {palette_missing[:16]}")
        print(
            "[prepare] restored texture_adapter from GAM checkpoint after pipe init "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
    elif args.use_texture_gate:
        print("[prepare] WARNING: use_texture_gate=1 but checkpoint has no texture_adapter state")
        raise RuntimeError("E2b requires texture_adapter gate parameters in GAM checkpoint")

    bf_state = gam_info.get("bf_state", None)
    if bf_state is not None and getattr(pipe, "bf_texture_conditioner", None) is not None:
        missing, unexpected = pipe.bf_texture_conditioner.load_state_dict(bf_state, strict=False)
        print(
            "[prepare] restored bf_texture_conditioner from GAM checkpoint after pipe init "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )

    palette_state = gam_state.get("palette_token_mlp", None)
    if palette_state and getattr(pipe, "palette_token_mlp", None) is None:
        pipe.palette_token_mlp = PaletteTokenMLP(
            cross_attention_dim=pipe.unet.config.cross_attention_dim,
            num_palette_tokens=args.num_palette_tokens,
        ).to(device=args.device, dtype=torch.float16)
    if palette_state and getattr(pipe, "palette_token_mlp", None) is not None:
        missing, unexpected = pipe.palette_token_mlp.load_state_dict(palette_state, strict=False)
        pipe.use_palette_tokens = bool(args.use_palette_tokens)
        print(
            "[prepare] restored palette_token_mlp from GAM checkpoint after pipe init "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
    elif args.use_palette_tokens and not palette_state:
        print("[prepare] WARNING: use_palette_tokens=1 but checkpoint has no palette_token_mlp state")

    tcpm_state = gam_state.get("tcpm_lite", None)
    if args.use_tcpm_lite:
        if tcpm_state:
            missing, unexpected = pipe.tcpm_lite.load_state_dict(tcpm_state, strict=False)
            print(
                "[prepare] restored tcpm_lite from GAM checkpoint "
                f"(missing={len(missing)}, unexpected={len(unexpected)})"
            )
        else:
            raise RuntimeError("use_tcpm_lite=1 but GAM checkpoint has no tcpm_lite state")

    pipe.effective_texture_num_tokens = args.texture_num_tokens
    if isinstance(pipe.texture_meta, dict):
        pipe.texture_meta.update(gam_meta)
    return pipe, generator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IMAGGarment')
    parser.add_argument('--GAM_model_ckpt',type=str)
    parser.add_argument('--prompt',type=str,default="A cloth")
    parser.add_argument('--sketch_path', type=str, required=True)
    parser.add_argument('--texture_path',type=str,required=True)
    parser.add_argument('--output_path', type=str, default="./output_sd_base")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--texture_ckpt',
        type=str,
        default="",
        help="Texture adapter checkpoint. If empty, GAM_model_ckpt is used.",
    )
    parser.add_argument('--guidance_scale', type=float, default=7.0)
    parser.add_argument('--sketch_scale', type=float, default=0.6)
    parser.add_argument('--ipa_scale', type=float, default=1.0)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--texture_mode', type=str, default='patch_resampled', choices=['patch_resampled', 'legacy_pooled'])
    parser.add_argument('--texture_num_tokens', type=int, default=16)
    parser.add_argument('--texture_scale', type=float, default=1.0)
    parser.add_argument('--texture_condition_mode', type=str, default='spatial', choices=['token', 'spatial', 'hybrid'])
    parser.add_argument('--layer_group_enabled', type=int, default=1, choices=[0, 1])
    parser.add_argument('--use_texture_gate', type=int, default=0, choices=[0, 1])
    parser.add_argument('--use_palette_tokens', type=int, default=0, choices=[0, 1])
    parser.add_argument('--num_palette_tokens', type=int, default=4)
    parser.add_argument('--palette_branch_scale_init', type=float, default=0.0)
    parser.add_argument('--gate_type', type=str, default='layer')
    parser.add_argument('--gate_init', type=str, default='identity')
    parser.add_argument('--gate_min', type=float, default=0.7)
    parser.add_argument('--gate_max', type=float, default=1.3)
    parser.add_argument('--use_balanced_fusion_gate', type=int, default=0, choices=[0, 1])
    parser.add_argument('--balanced_gate_hidden_dim', type=int, default=64)
    parser.add_argument('--balanced_gate_scale', type=float, default=0.2)
    parser.add_argument('--balanced_gate_min', type=float, default=0.8)
    parser.add_argument('--balanced_gate_max', type=float, default=1.2)
    parser.add_argument('--use_conflict_aware_gate', type=int, default=0, choices=[0, 1])
    parser.add_argument('--use_tcpm_lite', type=int, default=0, choices=[0, 1])
    parser.add_argument('--use_aa_tcr_fuse', type=int, default=0, choices=[0, 1])
    parser.add_argument('--tcpm_hidden_ratio', type=float, default=0.25)
    parser.add_argument('--tcpm_residual_scale_init', type=float, default=0.0)
    parser.add_argument('--conflict_texture_suppress_strength', type=float, default=0.1)
    parser.add_argument('--conflict_palette_suppress_strength', type=float, default=0.4)
    parser.add_argument('--conflict_deltae_norm', type=float, default=50.0)
    parser.add_argument('--conflict_threshold', type=float, default=0.70)
    parser.add_argument('--balanced_gate_trace_path', type=str, default="")
    parser.add_argument('--balanced_gate_trace_sample_id', type=str, default="")
    parser.add_argument(
        '--fusion_type',
        type=str,
        default='minimal',
        choices=['minimal', 'bfm_like'],
        help="Deprecated: decoupled spatial no longer uses fusion_type.",
    )
    parser.add_argument('--texture_preprocess_mode', type=str, default='crop_tile', choices=['plain_resize', 'crop_tile', 'plain'])
    parser.add_argument('--alpha1', type=float, default=2.0)
    parser.add_argument('--alpha2', type=float, default=2.0)
    parser.add_argument('--alpha3', type=float, default=1.5)
    parser.add_argument('--alpha4', type=float, default=1.0)
    parser.add_argument('--debug_spatial', action='store_true')

    parser.add_argument(
        '--base_model_path',
        type=str,
        default="auto",
        help=(
            "Base model path used to load tokenizer/text_encoder/unet for inference. "
            "Use 'auto' to read from GAM metadata when available."
        ),
    )
    parser.add_argument(
        '--vae_model_path',
        type=str,
        default="auto",
        help=(
            "VAE model path used for inference. "
            "Use 'auto' to read from GAM metadata when available."
        ),
    )
    parser.add_argument('--image_encoder_path', type=str, default='auto')
    parser.add_argument('--force_texture_num_tokens_override', action='store_true')
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help=(
            "Inference width. If omitted, use GAM checkpoint metadata width when available, otherwise fallback to 512."
        ),
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help=(
            "Inference height. If omitted, use GAM checkpoint metadata height when available, otherwise fallback to 640."
        ),
    )
    args = parser.parse_args()

    # save path
    output_path = args.output_path

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    pipe, generator = prepare(args)
    print('====================== pipe load finish ===================')
    _set_balanced_gate_trace(pipe, bool(args.balanced_gate_trace_path))

    num_samples = 1
    clip_image_processor = CLIPImageProcessor()

    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    
    #单图片
    prompt = args.prompt
    null_prompt = ''
    negative_prompt = ' worst quality, low quality'

    sketch_img = Image.open(args.sketch_path).convert("RGB").resize((args.width, args.height), Image.BILINEAR)
    vae_sketch = img_transform(sketch_img).unsqueeze(0)
    spatial_mask_img, spatial_mask_info = build_sketch_garment_mask(
        sketch_img, args.width, args.height
    )
    print(
        "spatial mask: "
        f"source={spatial_mask_info['mask_source']}, "
        f"confidence={spatial_mask_info['mask_confidence']:.4f}, "
        f"area={spatial_mask_info['mask_area_ratio']:.4f}, "
        f"fallback={spatial_mask_info['mask_fallback']}"
    )
    spatial_mask = transforms.ToTensor()(spatial_mask_img).unsqueeze(0)
    
    if args.texture_path is not None:
        texture_image = Image.open(args.texture_path).convert("RGB")
    else:
        texture_embeds = None
        texture_clip_image = None
    
    print(f"texture mode: {args.texture_mode}")
    print(f"fusion type: {args.fusion_type}")
    print(f"texture token count: {args.texture_num_tokens}")
    print(f"texture ckpt path: {args.texture_ckpt}")

    output = pipe(
        ref_image=vae_sketch,
        prompt=prompt,
        texture_clip_image=texture_image,
        texture_embeds=None,
        null_prompt=null_prompt,
        negative_prompt=negative_prompt,
        width=args.width,
        height=args.height,
        num_images_per_prompt=num_samples,
        guidance_scale=args.guidance_scale,
        sketch_scale=args.sketch_scale,
        ipa_scale=args.ipa_scale,
        generator=generator,
        num_inference_steps=args.num_inference_steps,
        texture_mode=args.texture_mode,
        texture_num_tokens=args.texture_num_tokens,
        texture_scale=args.texture_scale,
        texture_condition_mode=args.texture_condition_mode,
        use_palette_tokens=bool(args.use_palette_tokens),
        num_palette_tokens=args.num_palette_tokens,
        use_conflict_aware_gate=bool(args.use_conflict_aware_gate),
        conflict_texture_suppress_strength=args.conflict_texture_suppress_strength,
        conflict_palette_suppress_strength=args.conflict_palette_suppress_strength,
        conflict_deltae_norm=args.conflict_deltae_norm,
        conflict_threshold=args.conflict_threshold,
        fusion_type=args.fusion_type,
        texture_preprocess_mode=args.texture_preprocess_mode,
        alpha1=args.alpha1,
        alpha2=args.alpha2,
        alpha3=args.alpha3,
        alpha4=args.alpha4,
        spatial_mask=spatial_mask,
        debug_spatial=args.debug_spatial,
        force_texture_num_tokens_override=args.force_texture_num_tokens_override,
    )
    _save_balanced_gate_trace(
        pipe,
        args.balanced_gate_trace_path,
        sample_id=args.balanced_gate_trace_sample_id,
    )

    save_output = []
    save_output.append(output[0])
    save_output.insert(0, texture_image.resize((args.width, args.height), Image.BICUBIC))
    save_output.insert(0, sketch_img.resize((args.width, args.height), Image.BICUBIC))
    grid = image_grid(save_output, 1, 3)
    out_name = os.path.basename(args.sketch_path)
    grid.save(output_path + "/" + out_name)
    spatial_mask_img.save(output_path + "/" + os.path.splitext(out_name)[0] + "_mask.png")
    
    print(output_path + "/" + out_name)
