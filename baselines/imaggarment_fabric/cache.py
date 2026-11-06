"""一次编码冻结条件；缓存后验参数，训练时仍重新采样 VAE latent。"""
import os
import time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from diffusers import AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel, CLIPVisionModelWithProjection, CLIPImageProcessor
from .common import CONFIG, ROOT, read, write, output, sha


def image_tensor(path):
    size = (CONFIG['width'], CONFIG['height'])
    array = np.asarray(Image.open(path).convert('RGB').resize(size, Image.Resampling.BILINEAR), dtype=np.float32)
    return torch.from_numpy(array.transpose(2, 0, 1).copy()) / 127.5 - 1


class RawDataset(Dataset):
    def __init__(self, rows, indices, tokenizer):
        self.rows, self.indices, self.tokenizer = rows, indices, tokenizer
        # 官方 Color Adapter 也是默认 CLIPImageProcessor，不把面料先变为纯色。
        self.processor = CLIPImageProcessor()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        index = self.indices[index]
        row = self.rows[index]
        fabric = Image.open(row['fabric_path']).convert('RGB')
        return dict(index=index, target=image_tensor(row['target_path']), sketch=image_tensor(row['sketch_path']),
                    fabric=self.processor(images=fabric, return_tensors='pt').pixel_values[0],
                    text=self.tokenizer(row['caption'], max_length=77, padding='max_length', truncation=True,
                                        return_tensors='pt').input_ids[0])


def build(device, rank, world, short=False):
    cache_root = ROOT / 'cache' / 'preflight' if short else ROOT / 'cache'
    if (cache_root / 'complete.json').exists():
        print('CACHE_ALREADY_COMPLETE', flush=True)
        return
    started = time.time()
    rows = read(ROOT / 'datasets/train.json')
    if short:
        rows = rows[:16]
    count = len(rows)
    shapes = {key: (count, 4, CONFIG['height'] // 8, CONFIG['width'] // 8)
              for key in ['target_mean', 'target_std', 'sketch_mean', 'sketch_std']}
    shapes.update(text=(count, 77, 768), fabric=(count, 1024))
    if rank == 0:
        for key, shape in shapes.items():
            path = output(cache_root / (key + '.npy'))
            dtype = np.float32 if key.endswith(('_mean', '_std')) else np.float16
            np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=shape).flush()
    if world > 1:
        dist.barrier()
    arrays = {key: np.load(cache_root / (key + '.npy'), mmap_mode='r+') for key in shapes}
    base = ROOT / 'initialization/sd15'
    tokenizer = CLIPTokenizer.from_pretrained(base, subfolder='tokenizer', local_files_only=True)
    vae = AutoencoderKL.from_pretrained(base, subfolder='vae', local_files_only=True).to(device).eval().requires_grad_(False)
    text = CLIPTextModel.from_pretrained(base, subfolder='text_encoder', local_files_only=True).to(device, dtype=torch.bfloat16).eval().requires_grad_(False)
    vision = CLIPVisionModelWithProjection.from_pretrained(ROOT / 'initialization/image_encoder', local_files_only=True).to(device, dtype=torch.bfloat16).eval().requires_grad_(False)
    assert vision.config.projection_dim == 1024
    indices = list(range(count * rank // world, count * (rank + 1) // world))
    loader = DataLoader(RawDataset(rows, indices, tokenizer), batch_size=4, num_workers=2, pin_memory=True)
    with torch.inference_mode():
        for step, batch in enumerate(loader):
            idx = batch['index'].numpy()
            for key in ['target', 'sketch']:
                # VAE 保持 float32，缓存后验参数也保持 float32。
                posterior = vae.encode(batch[key].to(device)).latent_dist
                arrays[key + '_mean'][idx] = posterior.mean.float().cpu().numpy()
                arrays[key + '_std'][idx] = posterior.std.float().cpu().numpy()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                arrays['text'][idx] = text(batch['text'].to(device))[0].float().cpu().numpy()
                arrays['fabric'][idx] = vision(batch['fabric'].to(device)).image_embeds.float().cpu().numpy()
            if step % 100 == 0:
                print('CACHE', rank, step * 4, '/', len(indices), 'seconds', round(time.time() - started), flush=True)
        if rank == 0:
            ids = tokenizer('', max_length=77, padding='max_length', return_tensors='pt').input_ids.to(device)
            empty = text(ids)[0].float().cpu()
            torch.save(empty, output(cache_root / 'empty_text.pt'))
    for value in arrays.values():
        value.flush()
    if world > 1:
        dist.barrier()
    if rank == 0:
        write(cache_root / 'complete.json', dict(samples=count, seconds=time.time() - started,
            world=world, posterior='float32 mean/std; sampled afresh per exposure',
            text_and_fabric='bf16 frozen encoder output saved as float16',
            processor='official default CLIPImageProcessor: resize/center-crop 224, CLIP normalization',
            source_manifest_sha256=sha(ROOT / 'datasets/train.json')))


class CachedDataset:
    def __init__(self, short=False):
        cache_root = ROOT / 'cache' / 'preflight' if short else ROOT / 'cache'
        assert read(cache_root / 'complete.json')['samples'] == (16 if short else 45384)
        self.arrays = {key: np.load(cache_root / (key + '.npy'), mmap_mode='r')
                       for key in ['target_mean', 'target_std', 'sketch_mean', 'sketch_std', 'text', 'fabric']}
        self.empty = torch.load(cache_root / 'empty_text.pt', map_location='cpu', weights_only=True)

    def batch(self, indices, device):
        return {key: torch.from_numpy(np.array(value[indices], copy=True)).to(device=device, dtype=torch.float32)
                for key, value in self.arrays.items()}
