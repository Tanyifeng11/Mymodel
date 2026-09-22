"""完整texture模型的固定噪声旋转对照：24张，50步DDIM，无训练。"""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw
from tools.e14_pattern_probe import pixel_hash, write_json


def generate(args):
    import torch
    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
    from transformers import CLIPTokenizer, CLIPTextModel, CLIPImageProcessor
    from train_texture_adapter import TextureAdapter, IPAttnProcessor, AttnProcessor, load_image_encoder_flexible
    from models.bf_texture_module import BFTextureConditioner
    from checkpoint_utils import load_texture_warmstart
    from tools.e14_pattern_probe import training_preprocess, read_labels
    root, inputs = Path(args.output), Path(args.inputs)
    paths = {'baseline': Path(args.baseline),
             'legacy_pooled': Path(args.ab_root)/'legacy_pooled/checkpoint-final/pytorch_model.bin',
             'zero_final_tokens': Path(args.ab_root)/'zero_final_tokens/checkpoint-final/pytorch_model.bin'}
    for p in paths.values():
        if not p.is_file(): raise FileNotFoundError(str(p))
    rows = {r['sample_id']: r for r in read_labels(inputs/'feature_labels.csv')}
    selected = [rows['ref_%04d_%s' % (n,v)] for n in (13,30) for v in ('original','rot90')]
    for row in selected:
        with Image.open(inputs/row['texture']) as im:
            if pixel_hash(im.convert('RGB')) != row['pixel_sha256']: raise ValueError('参考hash不匹配')
    root.mkdir(parents=True,exist_ok=False)
    (root/'references').mkdir()
    for row in selected:
        with Image.open(inputs/row['texture']) as im: im.convert('RGB').save(root/'references'/(row['sample_id']+'.png'))
    device, dtype = args.device, torch.float16
    tokenizer=CLIPTokenizer.from_pretrained(args.base_model,subfolder='tokenizer',local_files_only=True)
    text=CLIPTextModel.from_pretrained(args.base_model,subfolder='text_encoder',local_files_only=True).to(device,dtype).eval()
    vision=load_image_encoder_flexible(args.clip_model,device,dtype).eval()
    vae=AutoencoderKL.from_pretrained(args.base_model,subfolder='vae',local_files_only=True).to(device,dtype).eval()
    processor=CLIPImageProcessor()
    ids=tokenizer(args.prompt,padding='max_length',max_length=tokenizer.model_max_length,truncation=True,return_tensors='pt').input_ids.to(device)
    with torch.inference_mode(): text_h=text(ids)[0]
    manifest=dict(complete=False,config=vars(args),records=[],
        protocol='DDIM eta=0; guidance_scale=1; no negative/unconditional branch; no sketch; fixed CPU noise per seed',
        limits=['仅texture预训练系统，不能直接代表E5联合模型。',
                '比较同seed下原图和旋转；方向判断需服装内部人工ROI，不能用全图轮廓替代纹理。'])
    write_json(root/'manifest.json',manifest)
    for stage,path in paths.items():
        torch.manual_seed(42)
        unet=UNet2DConditionModel.from_pretrained(args.base_model,subfolder='unet',local_files_only=True)
        procs={}
        for name in unet.attn_processors:
            if name.endswith('attn1.processor'):
                procs[name]=AttnProcessor();continue
            if name.startswith('mid_block'): hidden=unet.config.block_out_channels[-1]
            elif name.startswith('up_blocks'): hidden=list(reversed(unet.config.block_out_channels))[int(name.split('.')[1])]
            else: hidden=unet.config.block_out_channels[int(name.split('.')[1])]
            procs[name]=IPAttnProcessor(hidden_size=hidden,cross_attention_dim=unet.config.cross_attention_dim,num_tokens=16)
        unet.set_attn_processor(procs)
        bf=BFTextureConditioner(clip_embeddings_dim=vision.config.hidden_size,
                               cross_attention_dim=unet.config.cross_attention_dim,num_tokens=16)
        model=TextureAdapter(unet,torch.nn.ModuleList(unet.attn_processors.values()),bf)
        state=torch.load(str(path),map_location='cpu')
        filled=load_texture_warmstart(model,state)
        del state
        # 与训练相同保留BF的train路径（MHA实现路径），冻结全部参数。
        model.to(device=device,dtype=dtype).requires_grad_(False)
        model.unet.eval()
        for seed in (42,142):
            noise=torch.randn((1,4,64,48),generator=torch.Generator().manual_seed(seed))
            noise_hash=hashlib.sha256(noise.numpy().tobytes()).hexdigest()
            for row in selected:
                scheduler=DDIMScheduler.from_pretrained(args.base_model,subfolder='scheduler',local_files_only=True)
                scheduler.set_timesteps(args.steps,device=device)
                with Image.open(root/'references'/(row['sample_id']+'.png')) as im:
                    cnn,clip=training_preprocess(im.convert('RGB'),processor,384,512)
                with torch.inference_mode():
                    visual=vision(clip.to(device,dtype),output_hidden_states=True)
                    tokens=model.get_texture_condition_tokens(visual,cnn.to(device,dtype))
                    context=torch.cat([text_h,tokens],dim=1)
                    latents=noise.to(device,dtype)*scheduler.init_noise_sigma
                    for t in scheduler.timesteps:
                        prediction=unet(scheduler.scale_model_input(latents,t),t,encoder_hidden_states=context).sample
                        latents=scheduler.step(prediction,t,latents,eta=0).prev_sample
                    pixels=(vae.decode(latents/vae.config.scaling_factor).sample.float()/2+.5).clamp(0,1)
                    array=(pixels[0].permute(1,2,0).cpu().numpy()*255).round().astype('uint8')
                target=Path(stage)/('seed_%d'%seed)/(row['sample_id']+'.png')
                (root/target).parent.mkdir(parents=True,exist_ok=True)
                Image.fromarray(array).save(root/target)
                manifest['records'].append(dict(stage=stage,checkpoint=str(path),seed=seed,
                    sample_id=row['sample_id'],orientation=row['orientation'],image=str(target),
                    noise_sha256=noise_hash,reference_sha256=row['pixel_sha256'],filled_palette_keys=filled))
                write_json(root/'manifest.json',manifest)
                print(stage,seed,row['sample_id'],flush=True)
        del model,unet,bf,procs,context,tokens
        torch.cuda.empty_cache()
    manifest['complete']=True
    write_json(root/'manifest.json',manifest)
    report(root)


def report(root):
    root=Path(root);data=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if not data['complete'] or len(data['records'])!=24: raise ValueError('需要完整24张生成结果')
    lookup={(r['stage'],r['seed'],r['sample_id']):r for r in data['records']}
    review=[]
    for ref in (13,30):
        for seed in (42,142):
            canvas=Image.new('RGB',(3*384,2*540),'white');draw=ImageDraw.Draw(canvas)
            hashes=set()
            for col,stage in enumerate(['baseline','legacy_pooled','zero_final_tokens']):
                for j,variant in enumerate(['original','rot90']):
                    sid='ref_%04d_%s'%(ref,variant);r=lookup[stage,seed,sid];hashes.add(r['noise_sha256'])
                    with Image.open(root/r['image']) as im: canvas.paste(im,(col*384,j*540+28))
                    draw.text((col*384+5,j*540+5),stage+' '+variant,fill='black')
                    review.append(dict(stage=stage,seed=seed,sample_id=sid,image=r['image'],
                        garment_visible=None,pattern_visible=None,orientation='unreviewed',roi_xywh=None))
            if len(hashes)!=1: raise ValueError('配对噪声不一致')
            canvas.save(root/('comparison_ref_%04d_seed_%d.png'%(ref,seed)))
    review_path=root/'generation_review.json'
    if not review_path.exists(): write_json(review_path,review)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('run')
    for name in ['baseline','ab-root','inputs','base-model','clip-model','output']:q.add_argument('--'+name,required=True)
    q.add_argument('--device',default='cuda:0');q.add_argument('--steps',type=int,default=50)
    q.add_argument('--prompt',default='a sleeveless dress on a plain white background')
    q=sub.add_parser('report');q.add_argument('--root',required=True)
    a=p.parse_args()
    if a.command=='run':generate(a)
    else:report(a.root)
