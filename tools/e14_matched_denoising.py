"""固定目标、文本与噪声，比较清单配对参考、两种打乱参考及零 tokens。"""
import argparse
import hashlib
import json
import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from tools.e14_pattern_probe import write_json, pixel_hash


def choose_pairs(rows, indices, hashes):
    pairs={}
    for i in indices:
        candidates=[j for j in indices if j!=i and rows[j]['cloth']!=rows[i]['cloth'] and hashes[j]!=hashes[i]]
        if len(candidates)<2:raise ValueError('需要至少两个不同参考/目标来源')
        rng=random.Random(42000+i);rng.shuffle(candidates)
        a=candidates[0]
        b=next((j for j in candidates[1:] if hashes[j]!=hashes[a]),None)
        if b is None:raise ValueError('错误参考重复')
        pairs[i]=[a,b]
    return pairs


def summary(records):
    result={}
    for condition in ['wrong_1','wrong_2','zero_tokens']:
        samples={}
        for r in records:
            samples.setdefault(r['sample_index'],[]).append(r['losses'][condition]-r['losses']['matched'])
        values=[float(np.mean(v)) for v in samples.values()]
        result[condition]=dict(mean_loss_delta=float(np.mean(values)),median_sample_delta=float(np.median(values)),
            matched_better_samples=sum(v>0 for v in values),sample_count=len(values),
            per_sample={str(k):float(np.mean(v)) for k,v in samples.items()},
            by_timestep={str(t):float(np.mean([r['losses'][condition]-r['losses']['matched'] for r in records if r['timestep']==t]))
                         for t in sorted(set(r['timestep'] for r in records))})
    return result


def run(args):
    import torch
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTokenizer, CLIPTextModel
    from train_texture_adapter import MyDataset, TextureAdapter, IPAttnProcessor, AttnProcessor, load_image_encoder_flexible
    from models.bf_texture_module import BFTextureConditioner
    from checkpoint_utils import load_texture_warmstart
    device,dtype=args.device,torch.float16
    tokenizer=CLIPTokenizer.from_pretrained(args.base_model,subfolder='tokenizer',local_files_only=True)
    dataset=MyDataset(args.manifest,tokenizer,height=512,width=384,image_root_path=args.data_root,
                      texture_preprocess_mode='plain_resize',t_drop_rate=0,i_drop_rate=0,ti_drop_rate=0)
    if args.count<3 or args.count>len(dataset):raise ValueError('样本数不合法')
    # 固定随机抽样，不根据结果或caption挑选容易样本。
    indices=sorted(random.Random(42).sample(range(len(dataset)),args.count))
    hashes={}
    for i in indices:
        row=dataset.data[i]
        with Image.open(Path(args.data_root)/row.get('texture',row.get('color'))) as im:hashes[i]=pixel_hash(im.convert('RGB'))
    pairs=choose_pairs(dataset.data,indices,hashes)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);(out/'pairs').mkdir()
    scheduler=DDPMScheduler.from_pretrained(args.base_model,subfolder='scheduler',local_files_only=True)
    if scheduler.config.prediction_type!='epsilon':raise ValueError('当前训练损失要求epsilon预测')
    text=CLIPTextModel.from_pretrained(args.base_model,subfolder='text_encoder',local_files_only=True).to(device,dtype).eval()
    vision=load_image_encoder_flexible(args.clip_model,device,dtype).eval()
    vae=AutoencoderKL.from_pretrained(args.base_model,subfolder='vae',local_files_only=True).to(device,dtype).eval()
    unet=UNet2DConditionModel.from_pretrained(args.base_model,subfolder='unet',local_files_only=True)
    procs={}
    for name in unet.attn_processors:
        if name.endswith('attn1.processor'):procs[name]=AttnProcessor();continue
        if name.startswith('mid_block'):hidden=unet.config.block_out_channels[-1]
        elif name.startswith('up_blocks'):hidden=list(reversed(unet.config.block_out_channels))[int(name.split('.')[1])]
        else:hidden=unet.config.block_out_channels[int(name.split('.')[1])]
        procs[name]=IPAttnProcessor(hidden_size=hidden,cross_attention_dim=unet.config.cross_attention_dim,num_tokens=16)
    unet.set_attn_processor(procs)
    bf=BFTextureConditioner(clip_embeddings_dim=vision.config.hidden_size,cross_attention_dim=unet.config.cross_attention_dim,num_tokens=16)
    model=TextureAdapter(unet,torch.nn.ModuleList(unet.attn_processors.values()),bf)
    state=torch.load(args.checkpoint,map_location='cpu');filled=load_texture_warmstart(model,state);del state
    model.to(device=device,dtype=dtype).requires_grad_(False);unet.eval()
    tokens={};targets={};texts={}
    report=dict(complete=False,config=vars(args),records=[],samples=[],filled_palette_keys=filled,
        limits=['训练集清单配对不保证人工语义正确；不同路径/像素不保证独立布料。',
                '随机错误参考未匹配颜色，优势可能来自颜色；不能直接证明局部图案保真。',
                '全latent epsilon MSE不是生成质量或服装局部指标；无CFG、无sketch、不训练。',
                '每目标固定一次VAE后验采样；各条件共用目标、噪声、时间步和文本。',
                '按样本汇总；两个seed和多个时间步不是独立样本。'])
    with torch.inference_mode():
        for i in indices:
            batch=dataset[i]
            visual=vision(batch['clip_texture_image'].to(device,dtype),output_hidden_states=True)
            tokens[i]=model.get_texture_condition_tokens(visual,batch['texture_image'][None].to(device,dtype)).detach()
            texts[i]=text(batch['text_input_ids'].to(device))[0]
            posterior=vae.encode(batch['image'][None].to(device,dtype)).latent_dist
            eta=torch.randn(posterior.mean.shape,generator=torch.Generator().manual_seed(100000+i)).to(device,dtype)
            targets[i]=(posterior.mean+posterior.std*eta)*vae.config.scaling_factor
            if not torch.isfinite(targets[i]).all():raise ValueError('VAE输出非有限')
            row=dataset.data[i]
            report['samples'].append(dict(sample_index=i,cloth=row['cloth'],caption=row['caption'],
                matched_texture=row.get('texture',row.get('color')),reference_hash=hashes[i],wrong_indices=pairs[i]))
            canvas=Image.new('RGB',(4*192,280),'white');draw=ImageDraw.Draw(canvas)
            items=[('target',row['cloth'])]+[(label,dataset.data[j].get('texture',dataset.data[j].get('color')))
                  for label,j in [('matched',i),('wrong1',pairs[i][0]),('wrong2',pairs[i][1])]]
            for col,(label,path) in enumerate(items):
                with Image.open(Path(args.data_root)/path) as im:canvas.paste(im.convert('RGB').resize((192,256)),(col*192,24))
                draw.text((col*192+3,4),label,fill='black')
            canvas.save(out/'pairs'/('%05d.png'%i))
        del vae,vision,text
        torch.cuda.empty_cache()
        for i in indices:
            conditions={'matched':tokens[i],'wrong_1':tokens[pairs[i][0]],'wrong_2':tokens[pairs[i][1]],'zero_tokens':torch.zeros_like(tokens[i])}
            for seed in [42,142]:
                cpu_noise=torch.randn(targets[i].shape,generator=torch.Generator().manual_seed(seed+i*1000))
                noise=cpu_noise.to(device,dtype)
                for timestep in [1,181,481,781,981]:
                    t=torch.tensor([timestep],device=device,dtype=torch.long)
                    noisy=scheduler.add_noise(targets[i],noise,t)
                    losses={};reference=None;response={}
                    for name,token in conditions.items():
                        pred=unet(noisy,t,encoder_hidden_states=torch.cat([texts[i],token],1)).sample
                        if not torch.isfinite(pred).all():raise ValueError('预测非有限')
                        losses[name]=float((pred.float()-noise.float()).square().mean())
                        if reference is None:reference=pred.float()
                        response[name]=float((pred.float()-reference).square().mean().sqrt())
                    report['records'].append(dict(sample_index=i,seed=seed,timestep=timestep,losses=losses,
                        prediction_delta_rms=response,noise_sha256=hashlib.sha256(cpu_noise.numpy().tobytes()).hexdigest()))
            report['summary']=summary(report['records'])
            write_json(out/'denoising_report.json',report)
            print('completed',i,flush=True)
    report['complete']=True;write_json(out/'denoising_report.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['manifest','data-root','checkpoint','base-model','clip-model','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--count',type=int,default=32);p.add_argument('--device',default='cuda:0')
    run(p.parse_args())
