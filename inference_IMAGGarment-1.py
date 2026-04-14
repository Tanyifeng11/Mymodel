from pipelines.IMAGGarment_pipeline import IMAGGarment
import os
import torch

from PIL import Image
from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from torchvision import transforms
from transformers import CLIPImageProcessor
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from adapter.attention_processor import LogoCacheSAttnProcessor2_0, LogoRefSAttnProcessor2_0, LogoCacheCAttnProcessor2_0 , CAttnProcessor2_0,IPAttnProcessor2_0
import argparse


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


def prepare(args):
    generator = torch.Generator(device=args.device).manual_seed(42)
    
    #GAM data
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dtype=torch.float16, device=args.device)
    tokenizer = CLIPTokenizer.from_pretrained("SG161222/Realistic_Vision_V4.0_noVAE", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("SG161222/Realistic_Vision_V4.0_noVAE", subfolder="text_encoder").to(
        dtype=torch.float16, device=args.device)
    unet = UNet2DConditionModel.from_pretrained("SG161222/Realistic_Vision_V4.0_noVAE", subfolder="unet").to(
        dtype=torch.float16,device=args.device)
    image_encoder  = CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter",subfolder ="models/image_encoder").to(
        dtype=torch.float16, device=args.device)

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
            attn_procs[name] = IPAttnProcessor2_0( hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)

    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    adapter_modules = adapter_modules.to(dtype=torch.float16, device=args.device)
    del st
    

    ref_unet = UNet2DConditionModel.from_pretrained("SG161222/Realistic_Vision_V4.0_noVAE", subfolder="unet").to(
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
    model_sd = torch.load(args.GAM_model_ckpt, map_location="cpu", weights_only=False)["module"]

    ref_unet_dict = {}
    unet_dict = {}
    adapter_modules_dict = {}
    for k in model_sd.keys():
        if k.startswith("ref_unet"):
            ref_unet_dict[k.replace("ref_unet.", "")] = model_sd[k]
        elif k.startswith("unet"):
            unet_dict[k.replace("unet.", "")] = model_sd[k]
        elif k.startswith("adapter_modules"):
            adapter_modules_dict[k.replace("adapter_modules.", "")] = model_sd[k]
        else:
            print(k)

    ref_unet.load_state_dict(ref_unet_dict)
    adapter_modules.load_state_dict(adapter_modules_dict,strict=False)

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
                         scheduler=noise_scheduler,
                         safety_checker=StableDiffusionSafetyChecker,
                         feature_extractor=CLIPImageProcessor)
    return pipe, generator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IMAGGarment')
    parser.add_argument('--GAM_model_ckpt',type=str)
    parser.add_argument('--prompt',type=str,default="A cloth")
    parser.add_argument('--sketch_path', type=str, required=True)
    parser.add_argument('--texture_path',type=str,required=True)
    parser.add_argument('--output_path', type=str, default="./output_sd_base")
    parser.add_argument('--texture_ckpt', type=str, required=True)
    parser.add_argument('--guidance_scale', type=float, default=7.0)
    parser.add_argument('--sketch_scale', type=float, default=0.6)
    parser.add_argument('--ipa_scale', type=float, default=1.0)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--texture_mode', type=str, default='patch_resampled', choices=['patch_resampled', 'legacy_pooled'])
    parser.add_argument('--texture_num_tokens', type=int, default=16)
    parser.add_argument('--texture_scale', type=float, default=1.0)

    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--height",
        type=int,
        default=640,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
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
        transforms.Resize([640, 512], interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    
    #单图片
    prompt = args.prompt
    null_prompt = ''
    negative_prompt = ' worst quality, low quality'

    sketch_img = Image.open(args.sketch_path).convert("RGB")
    sketch_img = resize_img(sketch_img)
    vae_sketch = img_transform(sketch_img).unsqueeze(0)
    
    if args.texture_path is not None:
        texture_image = Image.open(args.texture_path)
    else:
        texture_embeds = None
        texture_clip_image = None
    
    print(f"texture mode: {args.texture_mode}")
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
    )

    save_output = []
    save_output.append(output[0])
    save_output.insert(0,texture_image.resize((512, 640), Image.BICUBIC))
    save_output.insert(0, sketch_img.resize((512, 640), Image.BICUBIC))
    grid = image_grid(save_output, 1, 3)
    grid.save(
        output_path + '/' + args.sketch_path.split("/")[-1])
    
    print( output_path + '/' + args.sketch_path.split("/")[-1])
