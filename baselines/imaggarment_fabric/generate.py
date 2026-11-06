"""GAM-only 三模态推理；生成清单中没有 GT 字段。"""
import copy
import hashlib
import time
import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from diffusers import AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel, CLIPVisionModelWithProjection, CLIPImageProcessor
from .common import CONFIG, ROOT, NAME, read, write, output, sha
from .cache import image_tensor
from .network import GarmentModel, scheduler, LogoRefSAttnProcessor2_0, IPAttnProcessor2_0


class Generator:
    def __init__(self, device, short=False):
        self.device = device
        self.model = GarmentModel('combined', device, training=False)
        self.weights = {}
        for stage in ['color', 'gam']:
            folder = 'preflight_' + stage if short else stage
            path = ROOT / 'training' / folder / 'final.pt'
            state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            assert state['stage'] == stage
            self.model.load_parameters(stage, state['parameters'])
            self.weights[stage] = dict(path=str(path), sha256=sha(path), updates=state['step'])
        self.model.reference_unet.to(dtype=torch.bfloat16)
        for proc in self.model.unet.attn_processors.values():
            if isinstance(proc, LogoRefSAttnProcessor2_0):
                proc.scale = CONFIG['sketch_scale']
            if isinstance(proc, IPAttnProcessor2_0):
                proc.scale = CONFIG['ipa_scale']
        base = ROOT / 'initialization/sd15'
        self.vae = AutoencoderKL.from_pretrained(base, subfolder='vae', local_files_only=True).to(device).eval().requires_grad_(False)
        self.text = CLIPTextModel.from_pretrained(base, subfolder='text_encoder', local_files_only=True).to(device, dtype=torch.bfloat16).eval().requires_grad_(False)
        self.tokenizer = CLIPTokenizer.from_pretrained(base, subfolder='tokenizer', local_files_only=True)
        self.vision = CLIPVisionModelWithProjection.from_pretrained(ROOT / 'initialization/image_encoder', local_files_only=True).to(device, dtype=torch.bfloat16).eval().requires_grad_(False)
        self.processor = CLIPImageProcessor()

    @torch.inference_mode()
    def image(self, row, destination):
        assert not any(key in row for key in ['gt', 'target_path', 'logo', 'mask'])
        assert sha(row['sketch']) == row['sketch_sha256'] and sha(row['reference']) == row['reference_sha256']
        started = time.time()
        torch.cuda.reset_peak_memory_stats(self.device)
        rng = torch.Generator(device=self.device).manual_seed(row['seed'])
        schedule = scheduler('inference')
        schedule.set_timesteps(CONFIG['steps'], device=self.device)
        latent = torch.randn((1, 4, CONFIG['height'] // 8, CONFIG['width'] // 8), generator=rng, device=self.device)
        latent_hash = hashlib.sha256(latent.float().cpu().numpy().tobytes()).hexdigest()
        latent *= schedule.init_noise_sigma
        sketch = self.vae.encode(image_tensor(row['sketch'])[None].to(self.device)).latent_dist.sample(generator=rng) * 0.18215
        ids = self.tokenizer([row['caption'], ''], max_length=77, padding='max_length', truncation=True, return_tensors='pt').input_ids.to(self.device)
        pixels = self.processor(images=Image.open(row['reference']).convert('RGB'), return_tensors='pt').pixel_values.to(self.device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            text, empty = self.text(ids)[0].chunk(2)
            embedding = self.vision(pixels).image_embeds
            conditional = torch.cat([text, self.model.image_proj(embedding)], dim=1)
            unconditional = torch.cat([empty, self.model.image_proj(torch.zeros_like(embedding))], dim=1)
            states = self.model.sketch_states(sketch)
            for t in schedule.timesteps:
                value = schedule.scale_model_input(latent, t)
                positive = self.model.unet(value, t, encoder_hidden_states=conditional,
                                           cross_attention_kwargs={'sa_hidden_states': states}).sample
                negative = self.model.unet(value, t, encoder_hidden_states=unconditional).sample
                prediction = negative + CONFIG['cfg'] * (positive - negative)
                latent = schedule.step(prediction, t, latent, eta=0, generator=rng).prev_sample
        decoded = self.vae.decode(latent.float() / 0.18215).sample.float().clamp(-1, 1)
        array = ((decoded[0].permute(1, 2, 0).cpu().numpy() / 2 + .5) * 255).round().clip(0, 255).astype('uint8')
        Image.fromarray(array).save(output(destination))
        self.model.clear_cache()
        meta = dict(row, arm=NAME, output_sha256=sha(destination), initial_latent_sha256=latent_hash,
                    native_width=CONFIG['width'], native_height=CONFIG['height'], steps=CONFIG['steps'],
                    cfg=CONFIG['cfg'], sketch_scale=CONFIG['sketch_scale'], ipa_scale=CONFIG['ipa_scale'],
                    checkpoints=self.weights, GT_used_in_inference=False, LEM=False,
                    seconds=time.time() - started, peak_gib=torch.cuda.max_memory_allocated(self.device) / 2**30,
                    sampler='official DDIM inference defaults; training GAM uses zero-terminal-SNR',
                    condition_encoder='official pooled CLIP + four tokens; raw RGB fabric')
        write(destination.with_suffix('.json'), meta)
        return array, meta


def generate(device, rank, world, short=False):
    if short and (ROOT / 'audit/smoke_complete.json').exists():
        assert read(ROOT / 'audit/smoke_complete.json')['passed']
        return
    generator = Generator(device, short=short)
    rows = read(ROOT / 'datasets/generation.json')
    if short:
        if rank != 0:
            return
        first = copy.deepcopy(rows[0])
        second = rows[1]
        conditions = [first, dict(first, reference=second['reference'], reference_sha256=second['reference_sha256']),
                      dict(first, sketch=second['sketch'], sketch_sha256=second['sketch_sha256'])]
        arrays, metadata = [], []
        for i, row in enumerate(conditions):
            array, meta = generator.image(row, ROOT / 'audit/smoke' / ('condition_%d.png' % i))
            arrays.append(array)
            metadata.append(meta)
        changes = dict(fabric_mean_absolute_change=float(np.abs(arrays[0].astype(float) - arrays[1]).mean()),
                       sketch_mean_absolute_change=float(np.abs(arrays[0].astype(float) - arrays[2]).mean()))
        assert all(value > 0 for value in changes.values()), changes
        write(ROOT / 'audit/smoke_complete.json', dict(passed=True, changes=changes, images=metadata,
                                                     purpose='condition wiring only; no quality or tuning conclusion'))
    else:
        for i, row in enumerate(rows):
            if i % world != rank:
                continue
            path = ROOT / 'outputs' / NAME / row['condition'] / (row['id'] + '.png')
            if path.exists() and path.with_suffix('.json').exists():
                assert read(path.with_suffix('.json'))['output_sha256'] == sha(path)
                continue
            generator.image(row, path)
            print('GENERATED', rank, i, row['id'], row['condition'], flush=True)
        if world > 1:
            dist.barrier()
        if rank == 0:
            assert all((ROOT / 'outputs' / NAME / row['condition'] / (row['id'] + '.png')).exists() for row in rows)
            write(ROOT / 'generation/complete.json', dict(passed=True, outputs=len(rows), checkpoints=generator.weights,
                                                         GT_used_in_inference=False))
