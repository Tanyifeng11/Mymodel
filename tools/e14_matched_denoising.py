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
    causal=getattr(args,'causal_suite',False)
    full=getattr(args,'complete_suite',False) or causal
    if full:
        from tools.e14_denoising_complete import color_pairs,FULL_STEPS,analyze
        args.regions=True
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
    colors=color_pairs(dataset.data,indices,hashes,args.data_root) if full else {}
    pair_review={};verified={}
    if causal:
        if not args.mask_root or not args.previous_report:raise ValueError('统一诊断需要已修正mask与前次report')
        from tools.e14_causal_suite import load_masks,configure_gate,verdict,generate_fixed,GENERATION_IDS
        from tools.e14_dropout_generation import install_source_balance
        if not all(i in indices for i in GENERATION_IDS):raise ValueError('固定生成样本不在清单中')
        if args.pair_review:
            pair_review=json.loads(Path(args.pair_review).read_text(encoding='utf-8'))
            for entry in pair_review.get('pairs',[]):
                if entry.get('status')!='verified_same_color_different_pattern':continue
                i,j=entry['sample_index'],entry['reference_index']
                if i not in indices or j not in indices or i==j or hashes[i]==hashes[j]:raise ValueError('已核验配对不合法')
                if not entry.get('reviewer') or not entry.get('notes'):raise ValueError('核验配对缺少审核人/依据')
                if entry.get('target_reference_hash')!=hashes[i] or entry.get('donor_reference_hash')!=hashes[j]:raise ValueError('审核图像hash不符')
                verified[i]=j
    previous=None
    if getattr(args,'previous_report',None):
        previous=json.loads(Path(args.previous_report).read_text(encoding='utf-8'))
        if not previous['complete']:raise ValueError('前次结果不完整')
        if [r['sample_index'] for r in previous['samples']]!=indices:raise ValueError('样本清单与前次不一致')
        for r in previous['samples']:
            i=r['sample_index'];row=dataset.data[i]
            if (r['reference_hash']!=hashes[i] or r['wrong_indices']!=pairs[i]
                or r['cloth']!=row['cloth'] or r['caption']!=row['caption']):
                raise ValueError('配对/数据与前次不一致')
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
    if causal:
        base_unet=UNet2DConditionModel.from_pretrained(args.base_model,subfolder='unet',local_files_only=True).to(device,dtype).eval().requires_grad_(False)
        causal_masks,corrected_parts=load_masks(args.mask_root,indices,out,device)
        (out/'token_cache').mkdir()
    balanced={};balanced_rotated={}
    tokens={};targets={};texts={};regions={};rotated={}
    spatial=getattr(args,'regions',False)
    if spatial:
        from tools.e14_denoising_regions import prepare,regional_summary
        (out/'error_maps').mkdir()
    report=dict(complete=False,config=vars(args),records=[],samples=[],filled_palette_keys=filled,
        limits=['训练集清单配对不保证人工语义正确；不同路径/像素不保证独立布料。',
                '随机错误参考未匹配颜色，优势可能来自颜色；不能直接证明局部图案保真。',
                '全latent epsilon MSE不是生成质量或服装局部指标；无CFG、无sketch、不训练。',
                '每目标固定一次VAE后验采样；各条件共用目标、噪声、时间步和文本。',
                '按样本汇总；两个seed和多个时间步不是独立样本。'])
    if causal:
        report['verified_pair_review']=dict(verified_count=len(verified),total=len(indices),
            status='available' if verified else 'unresolved_no_verified_pairs',review=pair_review)
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
            if full:
                from tools.e14_pattern_probe import training_preprocess
                with Image.open(Path(args.data_root)/row.get('texture',row.get('color'))) as im:
                    rotation=im.convert('RGB').transpose(Image.ROTATE_90)
                cnn,clip=training_preprocess(rotation,dataset.clip_image_processor,384,512)
                visual_rot=vision(clip.to(device,dtype),output_hidden_states=True)
                rotated[i]=model.get_texture_condition_tokens(visual_rot,cnn.to(device,dtype)).detach()
            if causal:
                handle=install_source_balance(bf.resampler,2.,.75)
                try:
                    balanced[i]=model.get_texture_condition_tokens(visual,batch['texture_image'][None].to(device,dtype)).detach()
                    balanced_rotated[i]=model.get_texture_condition_tokens(visual_rot,cnn.to(device,dtype)).detach()
                finally:handle.remove()
                np.savez_compressed(out/'token_cache'/('%05d.npz'%i),matched=tokens[i].float().cpu().numpy(),
                    rot90=rotated[i].float().cpu().numpy(),balanced=balanced[i].float().cpu().numpy(),balanced_rot90=balanced_rotated[i].float().cpu().numpy())
            report['samples'].append(dict(sample_index=i,cloth=row['cloth'],caption=row['caption'],
                matched_texture=row.get('texture',row.get('color')),reference_hash=hashes[i],wrong_indices=pairs[i]))
            if full:report['samples'][-1]['color_nearest']=colors[i]
            if spatial:
                if causal:
                    regions[i]=corrected_parts[i]
                    info=dict(mask_source='reviewed_113719',manual_review='assistant_coarse_not_ground_truth',
                        mask_low_confidence=False,latent_region_pixels={k:int(v.sum()) for k,v in regions[i].items()})
                else:regions[i],info=prepare(row,args.data_root,out,i,tuple(targets[i].shape[-2:]))
                report['samples'][-1]['region_diagnostics']=info
            canvas=Image.new('RGB',((6 if full else 4)*192,280),'white');draw=ImageDraw.Draw(canvas)
            items=[('target',row['cloth'])]+[(label,dataset.data[j].get('texture',dataset.data[j].get('color')))
                  for label,j in [('matched',i),('wrong1',pairs[i][0]),('wrong2',pairs[i][1])]]
            if full:
                donor=dataset.data[colors[i]['index']]
                items.append(('color_nearest',donor.get('texture',donor.get('color'))))
            for col,(label,path) in enumerate(items):
                with Image.open(Path(args.data_root)/path) as im:canvas.paste(im.convert('RGB').resize((192,256)),(col*192,24))
                draw.text((col*192+3,4),label,fill='black')
            if full:
                canvas.paste(rotation.resize((192,256)),(5*192,24));draw.text((5*192+3,4),'rot90',fill='black')
            canvas.save(out/'pairs'/('%05d.png'%i))
        del vision,text
        if not causal:del vae
        torch.cuda.empty_cache()
        for i in indices:
            conditions={'matched':tokens[i],'wrong_1':tokens[pairs[i][0]],'wrong_2':tokens[pairs[i][1]],'zero_tokens':torch.zeros_like(tokens[i])}
            if full:
                conditions.update(color_nearest=tokens[colors[i]['index']],rot90=rotated[i])
                if causal:
                    conditions.update(balanced=balanced[i],balanced_rot90=balanced_rotated[i],
                        region_all=tokens[i],region_window=tokens[i],base_unet=tokens[i])
                    if i in verified:conditions['verified_pattern']=tokens[verified[i]]
                if not all(bool(torch.isfinite(v).all()) for v in conditions.values()):raise ValueError('条件tokens非有限')
                report['samples'][indices.index(i)]['token_checks']={name:dict(
                    finite=bool(torch.isfinite(token).all()),
                    relative_change=float((token.float()-tokens[i].float()).norm()/tokens[i].float().norm().clamp_min(1e-12)))
                    for name,token in conditions.items()}
            for seed in [42,142]:
                cpu_noise=torch.randn(targets[i].shape,generator=torch.Generator().manual_seed(seed+i*1000))
                noise=cpu_noise.to(device,dtype)
                for timestep in (FULL_STEPS if full else [1,181,481,781,981]):
                    t=torch.tensor([timestep],device=device,dtype=torch.long)
                    noisy=scheduler.add_noise(targets[i],noise,t)
                    losses={};reference=None;response={};maps={}
                    for name,token in conditions.items():
                        gate=configure_gate(unet,causal_masks[i],name,timestep) if causal else None
                        pred=(base_unet(noisy,t,encoder_hidden_states=texts[i]).sample if causal and name=='base_unet'
                              else unet(noisy,t,encoder_hidden_states=torch.cat([texts[i],token],1)).sample)
                        if gate is not None and gate.calls!=len([p for p in unet.attn_processors.values() if hasattr(p,'to_k_ip')]):raise ValueError('区域mask未覆盖所有纹理层')
                        if not torch.isfinite(pred).all():raise ValueError('预测非有限')
                        losses[name]=float((pred.float()-noise.float()).square().mean())
                        if spatial:
                            maps[name]=(pred.float()-noise.float()).square().mean(1)[0].cpu().numpy()
                        if reference is None:reference=pred.float()
                        response[name]=float((pred.float()-reference).square().mean().sqrt())
                    repeat_error=None
                    if full and seed==42 and timestep==181:
                        if causal:configure_gate(unet,None,'matched',timestep)
                        repeat=unet(noisy,t,encoder_hidden_states=torch.cat([texts[i],tokens[i]],1)).sample
                        repeat_error=float((repeat.float()-reference).abs().max())
                    report['records'].append(dict(sample_index=i,seed=seed,timestep=timestep,losses=losses,
                        prediction_delta_rms=response,noise_sha256=hashlib.sha256(cpu_noise.numpy().tobytes()).hexdigest()))
                    if full:report['records'][-1]['matched_repeat_max_abs']=repeat_error
                    if causal:report['records'][-1]['region_window_expected_active']=141<=timestep<=261
                    if spatial:
                        filename='error_maps/%05d_seed%d_t%d.npz'%(i,seed,timestep)
                        np.savez_compressed(out/filename,**maps)
                        report['records'][-1]['error_maps']=filename
                        report['records'][-1]['region_losses']={
                            region:{name:float(error[mask].mean()) if mask.any() else None for name,error in maps.items()}
                            for region,mask in regions[i].items()}
            base_records=[r for r in report['records'] if r['timestep'] in [1,181,481,781,981]]
            report['summary']=summary(base_records)
            if spatial:report['region_summary']=regional_summary(base_records)
            write_json(out/'denoising_report.json',report)
            print('completed',i,flush=True)
    if previous is not None:
        old={(r['sample_index'],r['seed'],r['timestep']):r for r in previous['records']}
        errors=[]
        for r in report['records']:
            key=(r['sample_index'],r['seed'],r['timestep'])
            if key not in old:continue
            prior=old[key]
            if prior['noise_sha256']!=r['noise_sha256']:raise ValueError('噪声与前次不一致')
            errors.extend(abs(r['losses'][k]-prior['losses'][k]) for k in prior['losses'])
        report['previous_replay_max_abs_loss_difference']=max(errors)
        report['previous_replay_close']=max(errors)<1e-5
    report['complete']=True;write_json(out/'denoising_report.json',report)
    if full:analyze(out)
    if causal:
        verdict(out)
        from diffusers import DDIMScheduler
        del base_unet
        torch.cuda.empty_cache()
        with torch.inference_mode():generate_fixed(args,unet,vae,DDIMScheduler,texts,tokens,balanced,causal_masks,out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['manifest','data-root','checkpoint','base-model','clip-model','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--count',type=int,default=32);p.add_argument('--device',default='cuda:0')
    p.add_argument('--regions',action='store_true')
    p.add_argument('--previous-report')
    p.add_argument('--complete-suite',action='store_true')
    p.add_argument('--causal-suite',action='store_true')
    p.add_argument('--mask-root')
    p.add_argument('--pair-review')
    run(p.parse_args())
