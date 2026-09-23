"""固定 baseline 轨迹，检查纹理 tokens、attention 残差与 U-Net 预测响应。"""
import argparse
import hashlib
from pathlib import Path

from tools.e14_pattern_probe import write_json


def stats(value):
    x = value.detach().float()
    if not bool(x.isfinite().all()):
        raise ValueError('检测到非有限响应，停止汇总')
    return dict(rms=x.square().mean().sqrt().item(), max_abs=x.abs().max().item())


def difference(value, reference):
    x, y = value.detach().float(), reference.detach().float()
    result = stats(x - y)
    result['relative_l2'] = ((x-y).norm() / y.norm().clamp_min(1e-12)).item()
    result['exact_equal'] = bool(x.equal(y))
    return result


def run(args):
    import torch
    from PIL import Image
    from diffusers import DDIMScheduler, UNet2DConditionModel
    from transformers import CLIPTokenizer, CLIPTextModel, CLIPImageProcessor
    from train_texture_adapter import TextureAdapter, IPAttnProcessor, AttnProcessor, load_image_encoder_flexible
    from models.bf_texture_module import BFTextureConditioner
    from checkpoint_utils import load_texture_warmstart
    from tools.e14_pattern_probe import training_preprocess, read_labels, pixel_hash

    paths = {'baseline': Path(args.baseline),
             'legacy_pooled': Path(args.ab_root)/'legacy_pooled/checkpoint-final/pytorch_model.bin',
             'zero_final_tokens': Path(args.ab_root)/'zero_final_tokens/checkpoint-final/pytorch_model.bin'}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(str(path))
    inputs, root = Path(args.inputs), Path(args.output)
    rows = {r['sample_id']: r for r in read_labels(inputs/'feature_labels.csv')}
    names = ['ref_%04d_%s' % (n, v) for n in (13, 30) for v in ('original', 'rot90')]
    images = {}
    for name in names:
        with Image.open(inputs/rows[name]['texture']) as im:
            images[name] = im.convert('RGB')
        if pixel_hash(images[name]) != rows[name]['pixel_sha256']:
            raise ValueError('参考 hash 不匹配：' + name)
    root.mkdir(parents=True, exist_ok=False)
    device, dtype = args.device, torch.float16
    tokenizer = CLIPTokenizer.from_pretrained(args.base_model, subfolder='tokenizer', local_files_only=True)
    text = CLIPTextModel.from_pretrained(args.base_model, subfolder='text_encoder', local_files_only=True).to(device, dtype).eval()
    vision = load_image_encoder_flexible(args.clip_model, device, dtype).eval()
    processor = CLIPImageProcessor()
    ids = tokenizer(args.prompt, padding='max_length', max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors='pt').input_ids.to(device)
    with torch.inference_mode():
        text_h = text(ids)[0]
    del text
    report = dict(complete=False, config=vars(args), models={}, records=[],
                  protocol='同 seed 共用 baseline/ref_0013_original 的 50 步 DDIM 轨迹；CFG=1，无 sketch。',
                  limits=['残差差异包含上游 query 变化，不能视作该层独立因果贡献。',
                          '非零响应不等于正确图案跟随；此处只测试 texture 预训练系统。',
                          'relative_l2 在参考接近零时不稳定，需结合绝对 RMS。'])
    trajectories = {}
    selected_steps = {0, 10, 25, 40, 49}
    for stage, path in paths.items():
        torch.manual_seed(42)
        unet = UNet2DConditionModel.from_pretrained(args.base_model, subfolder='unet', local_files_only=True)
        procs = {}
        for name in unet.attn_processors:
            if name.endswith('attn1.processor'):
                procs[name] = AttnProcessor()
                continue
            if name.startswith('mid_block'):
                hidden = unet.config.block_out_channels[-1]
            elif name.startswith('up_blocks'):
                hidden = list(reversed(unet.config.block_out_channels))[int(name.split('.')[1])]
            else:
                hidden = unet.config.block_out_channels[int(name.split('.')[1])]
            procs[name] = IPAttnProcessor(hidden_size=hidden, cross_attention_dim=unet.config.cross_attention_dim, num_tokens=16)
        unet.set_attn_processor(procs)
        bf = BFTextureConditioner(clip_embeddings_dim=vision.config.hidden_size,
                                 cross_attention_dim=unet.config.cross_attention_dim, num_tokens=16)
        model = TextureAdapter(unet, torch.nn.ModuleList(unet.attn_processors.values()), bf)
        state = torch.load(str(path), map_location='cpu')
        filled = load_texture_warmstart(model, state)
        del state
        # 与上一轮生成一致，保留 BF 的 train 实现路径；全部参数冻结。
        model.to(device=device, dtype=dtype).requires_grad_(False)
        unet.eval()
        tokens = {}
        with torch.inference_mode():
            for name, im in images.items():
                cnn, clip = training_preprocess(im, processor, 384, 512)
                visual = vision(clip.to(device, dtype), output_hidden_states=True)
                tokens[name] = model.get_texture_condition_tokens(visual, cnn.to(device, dtype))
        anchor = names[0]
        tokens['zero_tokens'] = torch.zeros_like(tokens[anchor])
        ip = {n:p for n,p in unet.attn_processors.items() if isinstance(p, IPAttnProcessor)}
        report['models'][stage] = dict(checkpoint=str(path), filled_palette_keys=filled,
            layers=list(ip), token_stats={n:stats(v) for n,v in tokens.items()},
            token_differences={n:difference(v,tokens[anchor]) for n,v in tokens.items()})
        captured = {}

        def predict(x, t, condition, capture=True, scale=1.0):
            captured.clear()
            for name, proc in ip.items():
                proc.scale = scale
                proc.texture_probe_observer = ((lambda v, key=name: captured.__setitem__(key, v.detach().float().cpu())) if capture else None)
            with torch.inference_mode():
                out = unet(x, t, encoder_hidden_states=torch.cat([text_h, tokens[condition]], dim=1)).sample
            if capture and set(captured) != set(ip):
                raise RuntimeError('纹理残差回调缺失；请同步 adapter/attention_processor.py')
            return out.detach().float().cpu(), dict(captured)

        for seed in (42, 142):
            scheduler = DDIMScheduler.from_pretrained(args.base_model, subfolder='scheduler', local_files_only=True)
            if scheduler.config.prediction_type != 'epsilon':
                raise ValueError('当前检查要求 epsilon prediction，实际为 ' + scheduler.config.prediction_type)
            scheduler.set_timesteps(50, device=device)
            if stage == 'baseline':
                latent = torch.randn((1,4,64,48), generator=torch.Generator().manual_seed(seed)).to(device,dtype) * scheduler.init_noise_sigma
                trajectories[seed] = []
                for index, t in enumerate(scheduler.timesteps):
                    x = scheduler.scale_model_input(latent, t)
                    if index in selected_steps:
                        trajectories[seed].append((index, int(t.item()), x.detach().cpu().clone()))
                    prediction, _ = predict(x, t, anchor, capture=False)
                    latent = scheduler.step(prediction.to(device,dtype), t, latent, eta=0).prev_sample
            for index, timestep, cpu_x in trajectories[seed]:
                x = cpu_x.to(device,dtype)
                t = torch.tensor(timestep,device=device,dtype=torch.long)
                reference, ref_layers = predict(x,t,anchor)
                conditions = [(n,n,1.0) for n in tokens] + [('repeat',anchor,1.0),('scale_zero',anchor,0.0)]
                predictions = {}
                for label, condition, scale in conditions:
                    pred, layers = predict(x,t,condition,scale=scale)
                    predictions[label] = pred
                    report['records'].append(dict(stage=stage,seed=seed,step_index=index,timestep=timestep,
                        latent_sha256=hashlib.sha256(cpu_x.numpy().tobytes()).hexdigest(),condition=label,
                        prediction_stats=stats(pred), prediction_vs_original=difference(pred,reference),
                        layers={n:dict(residual=stats(v),vs_original=difference(v,ref_layers[n])) for n,v in layers.items()}))
                report['records'][-1]['zero_tokens_vs_scale_zero'] = difference(predictions['zero_tokens'],predictions['scale_zero'])
                # 直接比较每个参考自己的旋转对照，避免只看相对 0013 的差异。
                report['records'][-1]['rotation_pairs'] = {
                    str(n):difference(predictions['ref_%04d_rot90'%n],predictions['ref_%04d_original'%n]) for n in (13,30)}
                write_json(root/'noise_response.json',report)
                print(stage,seed,index,timestep,flush=True)
        for proc in ip.values():
            proc.texture_probe_observer = None
        del model,unet,bf,ip,tokens
        torch.cuda.empty_cache()
    report['complete'] = True
    write_json(root/'noise_response.json',report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['baseline','ab-root','inputs','base-model','clip-model','output']:
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--prompt',default='a sleeveless dress on a plain white background')
    run(parser.parse_args())
