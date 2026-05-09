from pipelines.IMAGGarment_pipeline import IMAGGarment
import os
import importlib
import importlib.util
from collections import deque
import torch
import numpy as np

from PIL import Image, ImageFilter
from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from torchvision import transforms
from transformers import CLIPImageProcessor
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from adapter.attention_processor import LogoCacheSAttnProcessor2_0, LogoRefSAttnProcessor2_0, LogoCacheCAttnProcessor2_0 , CAttnProcessor2_0,IPAttnProcessor2_0
from models.multiscale_texture_encoder import MultiScaleTextureEncoder
from models.spatial_injection import SpatialInjectionAdapter
import argparse

try:
    _repo_checkpoint_spec = importlib.util.find_spec("repo_utils.checkpoint_utils")
except ModuleNotFoundError:
    _repo_checkpoint_spec = None
_checkpoint_utils = importlib.import_module("repo_utils.checkpoint_utils") if _repo_checkpoint_spec is not None else importlib.import_module("checkpoint_utils")
load_checkpoint_file = _checkpoint_utils.load_checkpoint_file
detect_gam_checkpoint_format = _checkpoint_utils.detect_gam_checkpoint_format
infer_texture_num_tokens = _checkpoint_utils.infer_texture_num_tokens
extract_texture_metadata = _checkpoint_utils.extract_texture_metadata


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


def sketch_to_garment_mask(
    sketch: Image.Image,
    width: int,
    height: int,
    line_threshold: int = 245,
    dilate_size: int = 9,
) -> Image.Image:
    dilate_size = max(3, int(dilate_size) | 1)
    gray = sketch.convert("L").resize((width, height), Image.BILINEAR)
    line = np.asarray(gray) < line_threshold
    barrier = np.asarray(
        Image.fromarray((line.astype(np.uint8) * 255), mode="L").filter(
            ImageFilter.MaxFilter(dilate_size)
        )
    ) > 0

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

    return Image.fromarray((mask.astype(np.uint8) * 255), mode="L").filter(
        ImageFilter.MaxFilter(5)
    )


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

    generator = torch.Generator(device=args.device).manual_seed(42)
    resolved_image_encoder_path = resolve_image_encoder_path(args)
    print(f"[prepare] base model path: {args.base_model_path}")
    print(f"[prepare] vae model path: {args.vae_model_path}")
    print(f"[prepare] resolved image encoder path: {resolved_image_encoder_path}")
    
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
            attn_procs[name] = IPAttnProcessor2_0( hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, num_tokens=args.texture_num_tokens)

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
                         scheduler=noise_scheduler,
                         safety_checker=StableDiffusionSafetyChecker,
                         feature_extractor=CLIPImageProcessor)

    # IMAGGarment will load args.texture_ckpt in __init__, which can overwrite
    # adapter/BF states already loaded from GAM checkpoint. Restore GAM states here.
    gam_state = gam_info.get("state", {})
    if "texture_adapter" in gam_state:
        missing, unexpected = torch.nn.ModuleList(pipe.unet.attn_processors.values()).load_state_dict(
            gam_state["texture_adapter"], strict=False
        )
        print(
            "[prepare] restored texture_adapter from GAM checkpoint after pipe init "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )

    bf_state = gam_info.get("bf_state", None)
    if bf_state is not None and getattr(pipe, "bf_texture_conditioner", None) is not None:
        missing, unexpected = pipe.bf_texture_conditioner.load_state_dict(bf_state, strict=False)
        print(
            "[prepare] restored bf_texture_conditioner from GAM checkpoint after pipe init "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )

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
    spatial_mask_img = sketch_to_garment_mask(sketch_img, args.width, args.height)
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

    save_output = []
    save_output.append(output[0])
    save_output.insert(0, texture_image.resize((args.width, args.height), Image.BICUBIC))
    save_output.insert(0, sketch_img.resize((args.width, args.height), Image.BICUBIC))
    grid = image_grid(save_output, 1, 3)
    out_name = os.path.basename(args.sketch_path)
    grid.save(output_path + "/" + out_name)
    spatial_mask_img.save(output_path + "/" + os.path.splitext(out_name)[0] + "_mask.png")
    
    print(output_path + "/" + out_name)
