"""E5 生成阶段参考纹样响应：固定草图/文本/种子，仅替换参考图。

python -m tools.e14_generation_response run --help
python -m tools.e14_generation_response report --root PATH
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from tools.e14_controlled_patterns import pattern_mask
from tools.e14_pattern_probe import pixel_hash, write_json


VARIANTS = ['coarse_vertical', 'coarse_horizontal', 'fine_vertical', 'fine_horizontal', 'flat', 'repeat_coarse_vertical']


def references(root):
    root.mkdir()
    paths = {}
    for name, cycles in [('coarse', 3), ('fine', 12)]:
        mask = pattern_mask('stripe', 256, cycles, 0, 0)
        gray = np.where(mask, 32, 224).astype('uint8')
        for direction, array in [('vertical', gray), ('horizontal', np.rot90(gray))]:
            key = name + '_' + direction
            path = root / (key+'.png')
            Image.fromarray(array).convert('RGB').save(path)
            paths[key] = path
    paths['flat'] = root / 'flat.png'
    Image.new('RGB', (256,256), (176,176,176)).save(paths['flat'])
    paths['repeat_coarse_vertical'] = paths['coarse_vertical']
    return paths


def tiles_inside(mask, size=128):
    # 收缩边界后只取完整内部块，不把mask乘进FFT，避免mask边缘形成假条纹。
    interior = np.asarray(Image.fromarray((mask*255).astype('uint8')).filter(ImageFilter.MinFilter(9))) > 127
    h,w = interior.shape
    tiles = [(x,y,size,size) for y in range(0,h-size+1,size//2) for x in range(0,w-size+1,size//2)
             if interior[y:y+size,x:x+size].all()]
    return tiles


def spectrum(image, tiles):
    gray = np.asarray(image.convert('L'), dtype=np.float64)/255
    results=[]
    for x,y,w,h in tiles:
        patch=gray[y:y+h,x:x+w]
        patch=patch-patch.mean()
        power=np.abs(np.fft.fftshift(np.fft.fft2(patch*np.outer(np.hanning(h),np.hanning(w)))))**2
        ky,kx=np.meshgrid(np.fft.fftshift(np.fft.fftfreq(h)*h),np.fft.fftshift(np.fft.fftfreq(w)*w),indexing='ij')
        radius=np.sqrt(kx*kx+ky*ky)
        band=(radius>=1)&(radius<=min(h,w)/4)
        energy=float(power[band].sum())
        # 方向分数正值偏竖纹、负值偏横纹；零能量时指标不可定义。
        direction=float((power[band]*(kx[band]**2-ky[band]**2)/(radius[band]**2)).sum()/energy) if energy>1e-12 else None
        frequency=float((power[band]*radius[band]/w).sum()/energy) if energy>1e-12 else None
        results.append(dict(tile=[x,y,w,h], band_energy=energy, rms=float(np.sqrt(np.mean(patch**2))),
                            vertical_score=direction, frequency_cycles_per_pixel=frequency))
    def mean(key):
        v=[r[key] for r in results if r[key] is not None]
        return float(np.mean(v)) if v else None
    return dict(tile_count=len(tiles), vertical_score=mean('vertical_score'),
                frequency_cycles_per_pixel=mean('frequency_cycles_per_pixel'),
                rms=mean('rms'),band_energy=mean('band_energy'),tiles=results)


def run(args):
    root=Path(args.output).resolve()
    root.mkdir(parents=True,exist_ok=False)
    refs=references(root/'references')
    split=json.loads(Path(args.split).read_text(encoding='utf-8'))
    selected=[]
    for sid in args.sample_ids:
        selected.append(next(r for r in split if int(r['sample_id'])==sid))
    (root/'sketches').mkdir()
    manifest=dict(config={k:v for k,v in vars(args).items() if k!='func'},records=[],complete=False,
                  protocol='原始E5生成预处理；不应用rank_binary；不修改正式pipeline。',
                  reference_hashes={k:pixel_hash(Image.open(v).convert('RGB')) for k,v in refs.items()})
    for sample in selected:
        sid=str(sample['sample_id']).zfill(6)
        source=Path(args.data_root)/sample['sketch']
        sketch=root/'sketches'/f'{sid}.png'
        with Image.open(source) as image:
            image.convert('RGB').save(sketch)
        for seed in args.seeds:
            for variant in VARIANTS:
                dest=root/f'sample_{sid}'/f'seed_{seed}'/variant
                command=[sys.executable,'inference_IMAGGarment-1.py',
                    '--GAM_model_ckpt',args.checkpoint,'--texture_ckpt',args.texture_checkpoint,
                    '--image_encoder_path',args.clip_model,'--sketch_path',str(sketch),
                    '--texture_path',str(refs[variant]),'--output_path',str(dest),
                    '--prompt',args.prompt,'--seed',str(seed),'--device',args.device,
                    '--width','384','--height','512','--num_inference_steps','50',
                    '--guidance_scale','7','--sketch_scale','0.6','--texture_scale','1',
                    '--texture_condition_mode','token','--texture_mode','patch_resampled',
                    '--texture_preprocess_mode','plain_resize','--use_tcpm_lite','1',
                    '--use_texture_gate','1','--layer_group_enabled','1','--use_palette_tokens','0',
                    '--use_aa_tcr_fuse','0','--use_text_guided_resampler','0',
                    '--use_local_detail_adapter','0','--disable_nexus_adapter']
                manifest['records'].append(dict(sample_id=sid,seed=seed,variant=variant,
                    directory=str(dest.relative_to(root)), sketch=f'sketches/{sid}.png',
                    source_sketch=str(source),sketch_hash=pixel_hash(Image.open(sketch).convert('RGB')),
                    reference=str(refs[variant].relative_to(root)),command=command))
    write_json(root/'manifest.json',manifest)
    if args.prepare_only:
        print('仅准备输入和命令，尚未生成：',root)
        return
    for i,row in enumerate(manifest['records']):
        dest=root/row['directory']
        dest.mkdir(parents=True)
        print(f"[{i+1}/{len(manifest['records'])}] {row['directory']}",flush=True)
        with (dest/'inference.log').open('w',encoding='utf-8') as log:
            subprocess.run(row['command'],check=True,stdout=log,stderr=subprocess.STDOUT)
        # 推理入口保存三列图；输入草图使用PNG，避免生成图被JPEG压缩。
        with Image.open(dest/(row['sample_id']+'.png')) as grid:
            if grid.size!=(1152,512):
                raise ValueError('推理输出尺寸/布局与预期不符')
            grid.crop((768,0,1152,512)).save(dest/'generated.png')
    manifest['complete']=True
    write_json(root/'manifest.json',manifest)
    report(root)


def report(root):
    root=Path(root)
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    groups={}
    for row in manifest['records']:
        groups.setdefault((row['sample_id'],row['seed']),{})[row['variant']]=row
    results=[]
    for (sid,seed),records in groups.items():
        images={};masks={}
        for variant in VARIANTS:
            row=records[variant];dest=root/row['directory']
            images[variant]=Image.open(dest/'generated.png').convert('RGB')
            masks[variant]=np.asarray(Image.open(dest/f'{sid}_mask.png').convert('L'))>127
        mask=masks[VARIANTS[0]]
        if any(not np.array_equal(mask,m) for m in masks.values()):
            raise ValueError('同组固定草图mask不一致')
        tiles=tiles_inside(mask)
        if not tiles:
            tiles=tiles_inside(mask,size=64)
        metrics={k:spectrum(v,tiles) for k,v in images.items()}
        a=np.asarray(images['coarse_vertical'],dtype=float)/255
        b=np.asarray(images['repeat_coarse_vertical'],dtype=float)/255
        repeat_mse=float(np.mean((a-b)**2))
        diffs={}
        for kind,left,right,metric in [
            ('coarse_orientation','coarse_vertical','coarse_horizontal','vertical_score'),
            ('fine_orientation','fine_vertical','fine_horizontal','vertical_score'),
            ('vertical_frequency','fine_vertical','coarse_vertical','frequency_cycles_per_pixel'),
            ('horizontal_frequency','fine_horizontal','coarse_horizontal','frequency_cycles_per_pixel')]:
            l,r=metrics[left][metric],metrics[right][metric]
            diffs[kind]=None if l is None or r is None else l-r
        difference_from_flat={}
        interior=np.zeros(mask.shape,dtype=bool)
        for x,y,w,h in tiles:interior[y:y+h,x:x+w]=True
        flat=np.asarray(images['flat'],dtype=float)/255
        for k,image in images.items():
            difference_from_flat[k]=float(np.mean(((np.asarray(image,dtype=float)/255-flat)[interior])**2)) if interior.any() else None
        result=dict(sample_id=sid,seed=seed,repeat_exact=bool(np.array_equal(a,b)),repeat_mse=repeat_mse,
                    status='no_interior_tiles' if not tiles else ('repeat_mismatch' if repeat_mse else 'ready_for_review'),
                    metrics=metrics,signed_following_deltas=diffs,interior_mse_vs_flat=difference_from_flat)
        results.append(result)
        canvas=Image.new('RGB',(6*256,420),'white');draw=ImageDraw.Draw(canvas)
        for j,k in enumerate(VARIANTS):
            ref=Image.open(root/records[k]['reference']).convert('RGB').resize((128,128))
            canvas.paste(ref,(j*256,24));draw.text((j*256+2,4),k,fill='black')
            preview=images[k].copy();pd=ImageDraw.Draw(preview)
            for x,y,w,h in tiles:pd.rectangle((x,y,x+w-1,y+h-1),outline='red',width=1)
            canvas.paste(preview.resize((192,256)),(j*256,160))
        canvas.save(root/f'comparison_{sid}_seed{seed}.png')
    write_json(root/'response_report.json',dict(groups=results,limits=[
        '频谱指标只描述固定mask内部局部块；需查看红框是否位于真实生成服装内。',
        '方向差值/频率差值大于0才与预期方向一致，但不能仅凭符号判定成功。',
        '必须结合纹理能量、均匀灰对照、重复误差和两种子一致性；低能量下方向不可靠。',
        '固定mask来自草图，生成轮廓漂移时ROI可能失效；没有自动结构保真结论。',
        '不同种子只有2个、草图只有2张，属于探索性响应检查。',
        '优先128像素内部块，不足时退至64；比窗口更粗的条纹无法可靠估计频率，重叠块也非独立样本。',
        '参考是合成图，正常E5预处理可能有插值混杂；这是部署路径响应，不是受控输入探针。',
        '观察到响应不代表自然纹样保真；无响应也可能是合成参考域差异。']))
    print(root/'response_report.json',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('run')
    for name in ['split','data-root','checkpoint','texture-checkpoint','clip-model','output']:
        p.add_argument('--'+name,required=True)
    p.add_argument('--sample-ids',type=int,nargs='+',default=[2,18])
    p.add_argument('--seeds',type=int,nargs='+',default=[42,142])
    p.add_argument('--prompt',default='a garment')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--prepare-only',action='store_true')
    p.set_defaults(func=run)
    p=sub.add_parser('report');p.add_argument('--root',required=True)
    p.set_defaults(func=lambda args:report(args.root))
    args=parser.parse_args();args.func(args)


if __name__=='__main__':
    main()
