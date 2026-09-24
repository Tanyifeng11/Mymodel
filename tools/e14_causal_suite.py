"""E14统一因果对照辅助：区域干预、基座对照、离线判断和固定生成。"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from tools.e14_pattern_probe import write_json

VARIANTS=['matched','balanced','region_all','region_window']
GENERATION_IDS=[2082,44348]


class RegionGate:
    def __init__(self, mask, latent_hw=(64,48)):
        self.mask=mask
        self.sizes={ (latent_hw[0]//s)*(latent_hw[1]//s):(latent_hw[0]//s,latent_hw[1]//s) for s in [1,2,4,8]}
        self.calls=0

    def __call__(self,residual):
        import torch.nn.functional as F
        size=self.sizes.get(residual.shape[1])
        if size is None:raise ValueError('未知attention空间大小，不能静默跳过mask')
        mask=F.interpolate(self.mask.float(),size=size,mode='area').flatten(2).transpose(1,2)
        self.calls+=1
        return residual*mask.to(device=residual.device,dtype=residual.dtype)


def configure_gate(unet, mask, variant, timestep):
    gate=RegionGate(mask) if variant=='region_all' or (variant=='region_window' and 141<=int(timestep)<=261) else None
    for proc in unet.attn_processors.values():
        if hasattr(proc,'to_k_ip'):proc.texture_probe_transform=gate
    return gate


def load_masks(root, indices, output, device):
    import torch
    from tools.e14_denoising_regions import partition
    result={};parts={};provenance={}
    folder=Path(output)/'regions';folder.mkdir(exist_ok=True)
    import hashlib
    for i in indices:
        p=Path(root)/('%05d_mask.png'%i)
        with Image.open(p) as im:mask=im.convert('L')
        if mask.size!=(384,512):raise ValueError('需要已核验384x512 mask')
        mask.save(folder/p.name)
        parts[i]=partition(mask,(64,48))
        if not all(m.any() for m in parts[i].values()):raise ValueError('mask存在空区域')
        np.savez_compressed(folder/('%05d_masks.npz'%i),**parts[i])
        result[i]=torch.tensor(np.asarray(mask).copy(),device=device,dtype=torch.float32)[None,None]/255
        provenance[str(i)]=hashlib.sha256(p.read_bytes()).hexdigest()
    write_json(Path(output)/'mask_provenance.json',provenance)
    return result,parts


def verdict(root):
    """按原五步与预定异常窗口分别汇总，不自动宣布唯一根因。"""
    from tools.e14_denoising_complete import aggregate,BASE_STEPS
    root=Path(root);d=json.loads((root/'denoising_report.json').read_text(encoding='utf-8'))
    if not d['complete']:raise ValueError('推理未完成')
    report=dict(validation=dict(previous_replay_close=d.get('previous_replay_close')),
        comparisons={},pair_review=d.get('verified_pair_review',{}),
        limits=['正delta表示该干预比matched损失更高；不等于生成质量。',
                '基座U-Net与训练后模型不同，不能单凭损失差判定训练失败。',
                '区域mask来自已知目标，是oracle机制探针，不是部署可用性证明。',
                '未核验同色异图案时，该假设保持未解决；不以颜色近邻替代。',
                '同一32训练样本用于探索，不提供独立测试集泛化结论。'])
    for scope,steps in [('original_steps',BASE_STEPS),('window',[141,181,221,261])]:
        report['comparisons'][scope]={}
        for region in ['whole','interior','boundary','background']:
            rows=[r if region=='whole' else dict(r,losses=r['region_losses'][region]) for r in d['records'] if r['timestep'] in steps]
            # verified_pattern只在标注通过的配对上汇总，其余条件全样本。
            common=[dict(r,losses={k:v for k,v in r['losses'].items() if k!='verified_pattern'}) for r in rows]
            report['comparisons'][scope][region]=aggregate(common)
            verified=[r for r in rows if 'verified_pattern' in r['losses']]
            if verified:report['comparisons'][scope][region]['verified_pattern']=aggregate(verified)['verified_pattern']
    report['criteria']=dict(region='预先要求内部不退化、边界/背景下降，且观察逐样本一致性；再看固定生成。',
        representation='balanced需改善配对/方向响应并保留内部收益，不能仅凭tokens变大。',
        reference='verified_pattern仅人工核验有效，低覆盖保留不确定。')
    decisions={}
    for variant in ['balanced','region_all','region_window']:
        areas=report['comparisons']['original_steps']
        means={r:areas[r][variant]['mean_delta'] for r in ['interior','boundary','background']}
        decisions[variant]=dict(mean_deltas=means,
            mean_region_criterion_met=means['interior']<=0 and means['boundary']<0 and means['background']<0,
            note='仅预设均值筛查；不是显著性检验或生成通过结论')
    report['screening']=decisions
    rows=[dict(r,losses=dict(r['losses'],matched=r['losses']['base_unet'],trained_zero=r['losses']['zero_tokens']))
          for r in d['records'] if r['timestep'] in BASE_STEPS]
    report['trained_zero_minus_base_unet']=aggregate(rows)['trained_zero']
    write_json(root/'causal_summary.json',report)


def generate_fixed(args,unet,vae,scheduler_cls,texts,tokens,balanced,masks,output):
    import torch
    import hashlib
    out=Path(output)/'fixed_generation';out.mkdir(exist_ok=False)
    records=[]
    for i in GENERATION_IDS:
        for seed in [42,142]:
            cpu_noise=torch.randn((1,4,64,48),generator=torch.Generator().manual_seed(seed))
            for variant in VARIANTS:
                scheduler=scheduler_cls.from_pretrained(args.base_model,subfolder='scheduler',local_files_only=True)
                scheduler.set_timesteps(50,device=args.device)
                latent=cpu_noise.to(args.device,torch.float16)*scheduler.init_noise_sigma
                token=balanced[i] if variant=='balanced' else tokens[i]
                context=torch.cat([texts[i],token],1)
                for t in scheduler.timesteps:
                    gate=configure_gate(unet,masks[i],variant,t.item())
                    pred=unet(scheduler.scale_model_input(latent,t),t,encoder_hidden_states=context).sample
                    if gate is not None and gate.calls==0:raise ValueError('区域干预未接入')
                    latent=scheduler.step(pred,t,latent,eta=0).prev_sample
                image=(vae.decode(latent/vae.config.scaling_factor).sample.float()/2+.5).clamp(0,1)
                if not torch.isfinite(image).all():raise ValueError('生成非有限')
                filename='%05d_seed%d_%s.png'%(i,seed,variant)
                Image.fromarray((image[0].permute(1,2,0).cpu().numpy()*255).round().astype('uint8')).save(out/filename)
                records.append(dict(sample_index=i,seed=seed,variant=variant,image=filename,
                    noise_sha256=hashlib.sha256(cpu_noise.numpy().tobytes()).hexdigest()))
    configure_gate(unet,None,'matched',0)
    for i in GENERATION_IDS:
        for seed in [42,142]:
            canvas=Image.new('RGB',(384*4,540),'white');draw=ImageDraw.Draw(canvas)
            for col,variant in enumerate(VARIANTS):
                with Image.open(out/('%05d_seed%d_%s.png'%(i,seed,variant))) as im:canvas.paste(im,(384*col,28))
                draw.text((384*col+4,4),variant,fill='black')
            canvas.save(out/('%05d_seed%d_comparison.png'%(i,seed)))
    write_json(out/'manifest.json',dict(complete=True,records=records,protocol='16张；目标caption、CFG=1、50步DDIM；target mask oracle诊断'))
    write_json(out/'review.json',[dict(r,pattern_fidelity=None,background_damage=None,quality=None,review='pending') for r in records])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    verdict(p.parse_args().root)
