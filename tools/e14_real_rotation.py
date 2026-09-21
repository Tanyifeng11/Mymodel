"""真实条纹参考原图/90度旋转生成对照；16次生成，无模型训练。"""
import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from tools.e14_pattern_probe import pixel_hash, write_json
from tools.e14_generation_response import spectrum, tiles_inside


def rgb_counts(image):
    values, counts=np.unique(np.asarray(image).reshape(-1,3),axis=0,return_counts=True)
    return values,counts


def prepare(args):
    candidates=Path(args.candidates)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    with (candidates/'labels.csv').open(encoding='utf-8-sig',newline='') as f:
        labels=list(csv.DictReader(f))
    pairs=[];canvas=Image.new('RGB',(512,560),'white');draw=ImageDraw.Draw(canvas)
    for j,number in enumerate((13,30)):
        row=labels[number]
        if row['pattern']!='stripe' or row['confirmed']!='1':
            raise ValueError('候选标注与已核对条纹不符')
        original=Image.open(candidates/'thumbnails'/f'{number:04d}.png').convert('RGB')
        if pixel_hash(original)!=row['pixel_sha256']:
            raise ValueError('本地参考与原图像素哈希不符，不能替代原参考')
        rotated=original.transpose(Image.Transpose.ROTATE_90)
        a,b=rgb_counts(original),rgb_counts(rotated)
        assert all(np.array_equal(x,y) for x,y in zip(a,b))
        pair=dict(pair_id=f'ref_{number:04d}',sample_id=row['sample_id'],source_texture=row['texture'],
                  histogram_exact_equal=True,variants={})
        for k,image in [('original',original),('rot90',rotated)]:
            name=f'ref_{number:04d}_{k}.png';image.save(out/name)
            metrics=spectrum(image,[(0,0,image.width,image.height)])
            pair['variants'][k]=dict(file=name,pixel_sha256=pixel_hash(image),reference_metrics=metrics)
            x=0 if k=='original' else 256
            canvas.paste(image,(x,j*280+24));draw.text((x+4,j*280+4),name,fill='black')
        pairs.append(pair)
    split=json.loads(Path(args.split).read_text(encoding='utf-8'))
    sketches=[dict(sample_id=str(r['sample_id']).zfill(6),sketch=r['sketch']) for r in split if int(r['sample_id']) in (5,18)]
    if len(sketches)!=2:raise ValueError('未找到两张固定草图')
    write_json(out/'inputs.json',dict(pairs=pairs,sketches=sketches,seeds=[42,142],prompt='a garment',
        notes=['参考为256x256原像素，与原清单hash核对通过；旋转使用像素转置。',
               '草图使用服务器数据集原始文件；000005有褶皱线但无大块内部印花，000018为裙装轮廓。',
               '原图精确同色不保证模型缩放后直方图一致；不应用rank_binary。']))
    canvas.save(out/'reference_preview.png')
    print(out/'inputs.json')


def run(args):
    bundle=Path(args.inputs).resolve();root=Path(args.output).resolve()
    inputs=json.loads((bundle/'inputs.json').read_text(encoding='utf-8'))
    root.mkdir(parents=True,exist_ok=False)
    shutil.copytree(bundle,root/'inputs')
    (root/'sketches').mkdir()
    records=[]
    for sk in inputs['sketches']:
        sid=sk['sample_id'];path=Path(args.data_root)/sk['sketch']
        sketch=root/'sketches'/f'{sid}.png'
        with Image.open(path) as image:image.convert('RGB').save(sketch)
        for pair in inputs['pairs']:
            for seed in inputs['seeds']:
                for variant,item in pair['variants'].items():
                    ref=root/'inputs'/item['file']
                    with Image.open(ref) as image:
                        if pixel_hash(image.convert('RGB'))!=item['pixel_sha256']:raise ValueError('参考文件hash不符')
                    dest=root/f'sample_{sid}'/pair['pair_id']/f'seed_{seed}'/variant
                    cmd=[sys.executable,'inference_IMAGGarment-1.py','--GAM_model_ckpt',args.checkpoint,
                         '--texture_ckpt',args.texture_checkpoint,'--image_encoder_path',args.clip_model,
                         '--sketch_path',str(sketch),'--texture_path',str(ref),'--output_path',str(dest),
                         '--prompt',inputs['prompt'],'--seed',str(seed),'--device','cuda:0',
                         '--width','384','--height','512','--num_inference_steps','50',
                         '--guidance_scale','7','--sketch_scale','0.6','--texture_scale','1',
                         '--texture_condition_mode','token','--texture_mode','patch_resampled',
                         '--texture_preprocess_mode','plain_resize','--use_tcpm_lite','1',
                         '--use_texture_gate','1','--layer_group_enabled','1','--use_palette_tokens','0',
                         '--use_aa_tcr_fuse','0','--use_text_guided_resampler','0',
                         '--use_local_detail_adapter','0','--disable_nexus_adapter']
                    records.append(dict(sample_id=sid,pair_id=pair['pair_id'],seed=seed,variant=variant,
                                        directory=str(dest.relative_to(root)),command=cmd))
    manifest=dict(complete=False,records=records,config={k:v for k,v in vars(args).items() if k!='func'})
    write_json(root/'manifest.json',manifest)
    for i,row in enumerate(records):
        dest=root/row['directory'];dest.mkdir(parents=True)
        print(f'[{i+1}/{len(records)}] {row["directory"]}',flush=True)
        with (dest/'inference.log').open('w',encoding='utf-8') as log:
            subprocess.run(row['command'],check=True,stdout=log,stderr=subprocess.STDOUT)
        with Image.open(dest/(row['sample_id']+'.png')) as image:
            if image.size!=(1152,512):raise ValueError('推理输出布局不符')
            image.crop((768,0,1152,512)).save(dest/'generated.png')
    manifest['complete']=True;write_json(root/'manifest.json',manifest)
    report(root)


def paired_direction(reference_original,reference_rotated,generated_original,generated_rotated):
    values=[reference_original,reference_rotated,generated_original,generated_rotated]
    if any(v is None for v in values):return None
    # 两张参考的原始方向不同，必须按各自参考变化方向校正符号。
    expected=reference_original-reference_rotated
    return float(np.sign(expected)*(generated_original-generated_rotated)) if abs(expected)>1e-6 else None


def report(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    inputs=json.loads((root/'inputs/inputs.json').read_text(encoding='utf-8'))
    pairs={p['pair_id']:p for p in inputs['pairs']};groups={};results=[]
    for row in manifest['records']:
        groups.setdefault((row['sample_id'],row['pair_id'],row['seed']),{})[row['variant']]=row
    for (sid,pid,seed),rows in groups.items():
        images={};masks={}
        for name,row in rows.items():
            dest=root/row['directory'];images[name]=Image.open(dest/'generated.png').convert('RGB')
            masks[name]=np.asarray(Image.open(dest/f'{sid}_mask.png').convert('L'))>127
        if not np.array_equal(masks['original'],masks['rot90']):raise ValueError('同组草图mask不一致')
        tiles=tiles_inside(masks['original']) or tiles_inside(masks['original'],64)
        metrics={k:spectrum(v,tiles) for k,v in images.items()}
        ref=pairs[pid]['variants']
        score=paired_direction(*[ref[k]['reference_metrics']['vertical_score'] for k in ['original','rot90']],
                               *[metrics[k]['vertical_score'] for k in ['original','rot90']])
        results.append(dict(sample_id=sid,pair_id=pid,seed=seed,metrics=metrics,
                            signed_direction_following=score,status='review_required' if tiles else 'no_interior_tiles'))
        canvas=Image.new('RGB',(768,560),'white');draw=ImageDraw.Draw(canvas)
        for j,name in enumerate(['original','rot90']):
            canvas.paste(Image.open(root/'inputs'/ref[name]['file']).resize((192,192)),(j*384,24))
            draw.text((j*384+4,4),f'{pid} {name}',fill='black')
            preview=images[name].copy();pd=ImageDraw.Draw(preview)
            for x,y,w,h in tiles:pd.rectangle((x,y,x+w-1,y+h-1),outline='red',width=1)
            canvas.paste(preview.resize((240,320)),(j*384,230))
        canvas.save(root/f'comparison_{sid}_{pid}_seed{seed}.png')
    write_json(root/'rotation_report.json',dict(groups=results,limitations=[
        '正方向分数只说明平均轴向变化符合预期，不能单独视为复现90度旋转。',
        '必须结合ROI纹理能量和图片判断；褶皱、轮廓漂移及低纹理区域会干扰。',
        '两参考两草图两种子共8个配对，无新增重复组；此前重复一致性不能替代本次重复测量。',
        '图案尺度会因矩形输入缩放而受方向影响；本实验未隔离全部空间尺度混杂。']))
    print(root/'rotation_report.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('prepare')
    p.add_argument('--candidates',default='eval_outputs/e14_candidates/candidates')
    p.add_argument('--split',default='eval_outputs/full_condition_probe_6/112711/fixed_split.json')
    p.add_argument('--output',default='eval_outputs/e14_real_rotation_inputs');p.set_defaults(func=prepare)
    p=sub.add_parser('run')
    for name in ['inputs','output','data-root','checkpoint','texture-checkpoint','clip-model']:p.add_argument('--'+name,required=True)
    p.set_defaults(func=run)
    p=sub.add_parser('report');p.add_argument('--root',required=True);p.set_defaults(func=lambda a:report(a.root))
    args=parser.parse_args();args.func(args)


if __name__=='__main__':main()
