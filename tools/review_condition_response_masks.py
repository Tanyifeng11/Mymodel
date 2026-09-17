"""生成草图外轮廓核对图和画面边缘排除诊断；不把自动 mask 当真值。"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps


def review(run_dir, output_dir, sample_ids):
    root, out = Path(run_dir), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for sid in sample_ids:
        sid = f'{int(sid):06d}'
        matches = list((root / 'token').glob(f'{sid}_*'))
        if len(matches) != 1:
            raise ValueError(f'{sid} 样本目录不唯一')
        folder = matches[0]
        data = json.loads((folder/'condition_response_probe/probe.json').read_text(encoding='utf-8'))
        with Image.open(folder/f'comparison_{sid}.png') as image:
            comp = image.convert('RGB')
        w, h = comp.width // 3, comp.height
        sketch, gen = comp.crop((0, 0, w, h)), comp.crop((2*w, 0, 3*w, h))
        masks = list(folder.glob('*_mask.png'))
        if len(masks) != 1:
            raise ValueError(f'{sid} 推理 mask 不唯一')
        with Image.open(masks[0]) as image:
            mask = image.convert('L')
        kernel = data['region_kernel_input_pixels']
        radius = kernel // 2
        padded = ImageOps.expand(mask, border=radius, fill=0)
        crop = (radius, radius, radius+w, radius+h)
        outer = np.asarray(padded.filter(ImageFilter.MaxFilter(kernel)).crop(crop)) > 127
        inner = np.asarray(padded.filter(ImageFilter.MinFilter(kernel)).crop(crop)) > 127
        band = outer & ~inner
        canvas = Image.new('RGB', (3*w, h+26), 'white')
        draw = ImageDraw.Draw(canvas)
        for k, (image, region, label) in enumerate([
            (sketch, np.asarray(mask)>127, 'Sketch + mask'),
            (sketch, band, f'Sketch + {kernel}px outer boundary'),
            (gen, band, 'Generated + sketch boundary'),
        ]):
            array = np.array(image)
            array[region] = (array[region]*.55 + np.array([255,60,50])*.45).astype('uint8')
            canvas.paste(Image.fromarray(array), (k*w,26))
            draw.text((k*w+5,5), label, fill='black')
        canvas.save(out/f'{sid}.png')
        measurements = []
        for record in data['records']:
            step = record['step_index']
            with np.load(folder/f'condition_response_probe/step_{step:02d}.npz') as z:
                a, b = z['sketch_delta_h0.2'].astype('float64'), z['texture_delta_h0.2'].astype('float64')
                for margin in (0, 1, 2):
                    weight = z['weight_boundary'].copy()
                    if margin:
                        weight[:,:,:margin,:] = weight[:,:,-margin:,:] = 0
                        weight[:,:,:,:margin] = weight[:,:,:,-margin:] = 0
                    denom = np.sqrt((a*a*weight).sum()*(b*b*weight).sum())
                    measurements.append(dict(step=step, excluded_latent_border=margin,
                        excluded_image_border_y=margin*h/a.shape[-2],
                        excluded_image_border_x=margin*w/a.shape[-1],
                        cosine=float((a*b*weight).sum()/denom) if denom else None))
        results[sid] = dict(mask_info=data['metadata']['mask_info'], border_sensitivity=measurements)
    (out/'border_sensitivity.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--sample-ids', nargs='+', default=['18', '23'])
    args = parser.parse_args()
    review(args.run_dir, args.output_dir, args.sample_ids)
