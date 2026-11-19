"""衣内参考纹理监督；不对齐参考图与服装的像素位置。"""
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights


def structure_region(sketch, garment):
    body = -F.max_pool2d(-garment.float(), 9, stride=1, padding=4)
    gray = (((sketch.float() + 1) * .5) *
        sketch.new_tensor([.2989, .5870, .1140])[None, :, None, None]).sum(1, keepdim=True)
    kx = sketch.new_tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])[None, None]
    gx = F.conv2d(gray, kx.float(), padding=1)
    gy = F.conv2d(gray, kx.transpose(-1, -2).float(), padding=1)
    lines = ((gx.square() + gy.square() + 1e-6).sqrt() > .12).float()
    lines = F.max_pool2d(lines, 5, stride=1, padding=2)
    return garment * torch.maximum(1 - body, lines), body


def patch_coordinates(mask, side, maximum=4):
    occupancy = F.avg_pool2d(mask.float(), side, stride=side // 2)[0, 0]
    valid = (occupancy >= .99).nonzero(as_tuple=False)
    if len(valid) > maximum:
        selected = torch.linspace(0, len(valid) - 1, maximum, device=valid.device).round().long()
        valid = valid[selected]
    return [(int(x) * (side // 2), int(y) * (side // 2)) for y, x in valid]


def gray_contrast(patches):
    weights = patches.new_tensor([.2126, .7152, .0722])[None, :, None, None]
    gray = (patches.float() * weights).sum(1, keepdim=True)
    mean = gray.mean((2, 3), keepdim=True)
    std = gray.std((2, 3), keepdim=True, unbiased=False).clamp_min(4 / 255)
    # 只让新增目标学习灰度图案；颜色继续由现有 RGB 目标负责。
    return ((gray - mean) / std * .36).clamp(-1, 1).repeat(1, 3, 1, 1)


def contextual_distance(a, b, bandwidth=.5):
    def directed(x, y):
        center = y.mean(0, keepdim=True)
        x = F.normalize(x - center, dim=1, eps=1e-6)
        y = F.normalize(y - center, dim=1, eps=1e-6)
        distance = (1 - x @ y.T).clamp_min(0)
        relative = distance / (distance.min(1, keepdim=True).values + 1e-5)
        affinity = torch.softmax((1 - relative) / bandwidth, dim=1)
        return -affinity.max(0).values.mean().clamp_min(1e-8).log()
    return .5 * (directed(a.float(), b.float()) + directed(b.float(), a.float()))


class ReferenceRegionLoss(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = VGG19_Weights.DEFAULT if pretrained else None
        self.features = vgg19(weights=weights).features[:27].eval().requires_grad_(False)
        self.register_buffer('mean', torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer('std', torch.tensor([.229, .224, .225])[None, :, None, None])

    def extract(self, image, layers=(17, 26), normalized=True):
        x = image.float()
        if normalized:
            x = (((x + 1) * .5).clamp(0, 1) - self.mean) / self.std
        outputs = []
        for index, layer in enumerate(self.features):
            x = layer(x)
            if index in layers:
                outputs.append(x)
            if index == max(layers):
                break
        return outputs

    @staticmethod
    def gram(features):
        b, c, h, w = features.shape
        flat = features.flatten(2)
        return flat @ flat.transpose(1, 2) / (c * h * w + 1e-6)

    def gt_gram(self, prediction, target, mask, normalized=True):
        # 保留原 relu3_4/relu4_4 和 L1 Gram，仅统一输入预处理。
        pred = self.extract(prediction * mask, normalized=normalized)
        with torch.no_grad():
            truth = self.extract(target * mask, normalized=normalized)
        return sum(F.l1_loss(self.gram(a), self.gram(b)) for a, b in zip(pred, truth))

    def reference_texture(self, prediction, reference, body):
        # 特征值保持原图上下文，但新增纹理梯度严格限制在衣内。
        prediction = prediction * body + prediction.detach() * (1 - body)
        losses, counts = [], []
        for i in range(len(prediction)):
            generated, refs = [], []
            for side in (32, 64):
                for x, y in patch_coordinates(body[i:i+1], side):
                    patch = prediction[i:i+1, :, y:y+side, x:x+side]
                    generated.append(F.interpolate(patch, (64, 64), mode='bilinear', align_corners=False))
                h, w = reference.shape[-2:]
                for y in (0, (h-side)//2, h-side):
                    for x in (0, (w-side)//2, w-side):
                        patch = reference[i:i+1, :, y:y+side, x:x+side]
                        refs.append(F.interpolate(patch, (64, 64), mode='bilinear', align_corners=False))
            counts.append(len(generated))
            if not generated:
                continue
            generated = gray_contrast(torch.cat(generated))
            refs = gray_contrast(torch.cat(refs))
            pred_features = self.extract(generated, layers=(8, 13))
            with torch.no_grad():
                ref_features = self.extract(refs, layers=(8, 13))
            for a, b in zip(pred_features, ref_features):
                a = a.permute(0, 2, 3, 1).reshape(-1, a.shape[1])
                b = b.permute(0, 2, 3, 1).reshape(-1, b.shape[1])
                # 确定性采样，不消耗训练噪声的私有 RNG，保证两组输入流相同。
                a = a[torch.linspace(0, len(a)-1, min(256, len(a)), device=a.device).round().long()]
                b = b[torch.linspace(0, len(b)-1, min(512, len(b)), device=b.device).round().long()]
                losses.append(contextual_distance(a, b))
        loss = torch.stack(losses).mean() if losses else prediction.sum() * 0
        return loss, counts


def training_losses(pipe, network, batch, rng, style, arm, weights, audit_mode=False):
    from diffusers import DDIMScheduler
    from tools.bf_t3_frozen_core import sketch_states, digest
    from train_GAM_texture_joint import (masked_edge_l1, texture_color_stat_loss,
        build_region_masks, masked_l1_loss)
    scheduler = DDIMScheduler(beta_start=.00085, beta_end=.012,
        beta_schedule='scaled_linear', num_train_timesteps=1000)
    device = batch['vae_cloth'].device
    with torch.no_grad():
        posterior = pipe.vae.encode(batch['vae_cloth']).latent_dist
        latent = (posterior.mean + posterior.std * torch.randn(posterior.mean.shape,
            device=device, generator=rng)) * pipe.vae.config.scaling_factor
        sketch_latent = pipe.vae.encode(batch['vae_sketch']).latent_dist.mean * pipe.vae.config.scaling_factor
        drop = float(torch.rand((), device=device, generator=rng))
        if audit_mode:
            drop = 0.
        text_ids = batch['input_ids']
        if drop < .2 or .25 <= drop < .3:
            text_ids = pipe.tokenizer('', padding='max_length', truncation=True,
                max_length=77, return_tensors='pt').input_ids.to(device)
        text = pipe.text_encoder(text_ids)[0]
        noise = torch.randn(latent.shape, device=device, dtype=latent.dtype, generator=rng)
        timestep = torch.randint(0, 1000, (len(latent),), device=device, generator=rng)
        if audit_mode:
            timestep.fill_(100)
        states = sketch_states(pipe, sketch_latent.to(pipe.reference_unet.dtype), timestep)
    reference = batch['texture_image']
    keep = not (.2 <= drop < .3)
    noisy = scheduler.add_noise(latent, noise, timestep)
    predicted = network(noisy, timestep, text, batch['vae_sketch'],
        reference if keep else torch.zeros_like(reference), states)
    if audit_mode:
        predicted = predicted.detach().requires_grad_(True)
    mse = F.mse_loss(predicted.float(), noise.float())
    alpha = scheduler.alphas_cumprod.to(device)[timestep].reshape(-1, 1, 1, 1)
    x0 = (noisy.float() - (1-alpha).sqrt() * predicted.float()) / alpha.sqrt()
    decoded = pipe.vae.decode(x0 / pipe.vae.config.scaling_factor).sample.float()
    mask = batch['garment_mask']
    structural, body = structure_region(batch['vae_sketch'], mask)
    gram = style.gt_gram(decoded, batch['vae_cloth'], mask)
    edge = masked_edge_l1(decoded, batch['vae_sketch'], structural if arm == 'texture' else mask)
    color = texture_color_stat_loss(decoded, reference, mask) if keep else predicted.sum() * 0
    body_rgb = texture_color_stat_loss(decoded, reference, body) if keep else predicted.sum() * 0
    # 原外溢控制项和系数保持：边界及背景仍对齐 GT。
    rest = masked_l1_loss(decoded, batch['vae_cloth'], 1-body)
    texture, patch_counts = predicted.sum() * 0, [0] * len(predicted)
    if keep and (arm == 'texture' or audit_mode) and int(timestep.max()) <= 400:
        texture, patch_counts = style.reference_texture(decoded, reference, body)
    total = (mse + weights['gram'] * gram + .05 * edge + .1 * color
        + .1 * body_rgb + .03816244801608925 * rest
        + (weights['reference'] * texture if arm == 'texture' else 0))
    raw = dict(total=total, mse=mse, gram=gram, edge=edge, color=color,
        body=body_rgb, rest=rest, reference_texture=texture)
    token = dict(index=int(batch['dataset_index'][0]), t=int(timestep[0]),
        drop=drop, noise_sha256=digest(noise))
    audit = dict(predicted=predicted, decoded=decoded, patch_counts=patch_counts)
    return raw, token, audit
