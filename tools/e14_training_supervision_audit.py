"""审计真实预训练数据路径及损失敏感性；不训练模型，不自动判定配对正确。"""
import argparse
import ast
import hashlib
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw


def training_classes():
    # 直接执行原文件的两个类定义，避免导入整个训练入口及无关模型依赖。
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms
    from torchvision.models import vgg19, VGG19_Weights
    from transformers import CLIPImageProcessor
    from texture_preprocess import preprocess_texture_image
    import os
    source = Path(__file__).resolve().parents[1] / 'train_texture_adapter.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in ('MyDataset', 'VGGStyleLoss')]
    env = dict(torch=torch, nn=nn, F=F, transforms=transforms, vgg19=vgg19,
               VGG19_Weights=VGG19_Weights, CLIPImageProcessor=CLIPImageProcessor,
               preprocess_texture_image=preprocess_texture_image, os=os, json=json,
               random=random, Image=Image)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), env)
    return env['MyDataset'], env['VGGStyleLoss']


class DummyTokenizer:
    model_max_length = 77

    def __call__(self, *args, **kwargs):
        import torch
        return SimpleNamespace(input_ids=torch.zeros(1, 77, dtype=torch.long))


def select_rows(rows, count, seed):
    # 文本只用于召回条纹候选，不能充当图案真值；另外抽取随机训练样本。
    stripe = [i for i, r in enumerate(rows) if 'strip' in r.get('caption', '').lower()]
    rng = random.Random(seed)
    rng.shuffle(stripe)
    chosen = stripe[:count // 2]
    rest = [i for i in range(len(rows)) if i not in chosen]
    rng.shuffle(rest)
    return chosen + rest[:max(0, count-len(chosen))]


def run(args):
    import torch
    from torchvision import transforms
    Dataset, StyleLoss = training_classes()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    dataset = Dataset(args.manifest, DummyTokenizer(), width=args.width, height=args.height,
                      image_root_path=args.data_root, texture_preprocess_mode=args.mode,
                      t_drop_rate=0, i_drop_rate=0, ti_drop_rate=0)
    loss_fn = None
    if args.vgg:
        from torchvision.models import VGG19_Weights
        filename = VGG19_Weights.DEFAULT.url.rsplit('/', 1)[-1]
        cached = Path(torch.hub.get_dir()) / 'checkpoints' / filename
        if not cached.is_file():
            raise FileNotFoundError('不自动下载VGG；请提供已有缓存：' + str(cached))
        loss_fn = StyleLoss().to(args.device).eval()
    report = dict(config=vars(args), samples=[], code_sha256={},
        limits=['当前代码及指定参数的重放，不证明历史启动参数；核对服务器原训练日志。',
                '方向指标和caption均不能自动判定服装/参考配对；请在review.json填写人工结果。',
                'VGG测试是参考自对照的损失敏感性，不是生成样本损失或梯度贡献。',
                'Gram对特征空间位置置换不变，不意味着VGG对输入旋转严格不变。',
                '去噪损失监督cloth目标；其条件依赖程度需要模型实验，本工具不据此断言忽略参考。'])
    repo = Path(__file__).resolve().parents[1]
    for name in ['train_texture_adapter.py', 'texture_preprocess.py']:
        report['code_sha256'][name] = hashlib.sha256((repo/name).read_bytes()).hexdigest()
    to_pil = transforms.ToPILImage()
    review = []
    for idx in select_rows(dataset.data, args.count, args.seed):
        random.seed(args.seed + idx)
        row, batch = dataset.data[idx], dataset[idx]
        folder = out / ('sample_%05d' % idx)
        folder.mkdir()
        raw = Image.open(Path(args.data_root) / row.get('texture', row.get('color'))).convert('RGB')
        cloth = Image.open(Path(args.data_root) / row['cloth']).convert('RGB')
        cond = to_pil((batch['texture_image']*.5+.5).clamp(0, 1))
        processor = dataset.clip_image_processor
        train_clip = batch['clip_texture_image']
        probe_clip = processor(images=raw, return_tensors='pt').pixel_values
        def clip_image(t):
            mean = torch.tensor(processor.image_mean).view(3, 1, 1)
            std = torch.tensor(processor.image_std).view(3, 1, 1)
            return to_pil((t[0]*std+mean).clamp(0, 1))
        images = [('cloth', cloth), ('raw_texture', raw), ('train_cnn', cond),
                  ('train_clip', clip_image(train_clip)), ('previous_probe_clip', clip_image(probe_clip))]
        preview = Image.new('RGB', (256*len(images), 282), 'white')
        draw = ImageDraw.Draw(preview)
        for col, (name, im) in enumerate(images):
            im.save(folder/(name+'.png'))
            thumb = im.copy(); thumb.thumbnail((256, 256))
            preview.paste(thumb, (col*256, 0))
            draw.text((col*256+4, 260), name, fill='black')
        preview.save(folder/'comparison.png')
        tensor = batch['texture_image'].unsqueeze(0).to(args.device)
        # 旋转在原图执行，再走同一尺寸变换；不是旋转非方形tensor后强行比较。
        rotated = dataset.transform(raw.transpose(Image.Transpose.ROTATE_90)).unsqueeze(0).to(args.device)
        target = batch['texture_ref'].unsqueeze(0).to(args.device)
        with torch.inference_mode():
            item = dict(index=idx, manifest_row=row,
                clip_input_mae=float((train_clip-probe_clip).abs().mean()),
                cnn_condition_vs_raw_target_mae=float((tensor-target).abs().mean()),
                rgb_mean_cosine_rotation_loss=float(1-torch.nn.functional.cosine_similarity(
                    target.mean((2,3)), rotated.mean((2,3)), dim=1).mean()))
            if loss_fn is not None:
                item['vgg_gram'] = dict(identity=float(loss_fn(target,target)),
                    rotated_reference=float(loss_fn(target,rotated)),
                    conditioned_vs_raw=float(loss_fn(tensor,target)))
        report['samples'].append(item)
        review.append(dict(index=idx, preview=str((folder/'comparison.png').relative_to(out)),
            pattern_visible=None, same_pattern=None, direction_relation='unreviewed',
            cloth_pattern_roi_xywh=None, notes=''))
        print(idx, 'CLIP MAE', item['clip_input_mae'], flush=True)
    for name, data in [('audit.json',report), ('review.json',review)]:
        (out/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--mode', choices=['plain_resize','crop_tile'], default='plain_resize')
    p.add_argument('--width', type=int, default=384)
    p.add_argument('--height', type=int, default=512)
    p.add_argument('--count', type=int, default=40)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--vgg', action='store_true')
    p.add_argument('--device', default='cpu')
    run(p.parse_args())
