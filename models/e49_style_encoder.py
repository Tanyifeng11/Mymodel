"""StyleAdapter 的编码消融：修改编码输入，不修改 E5 共享的 CLIP 模块。"""
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from adapter.resampler import PerceiverResampler
from models.e48_reference_attention import IndependentReferenceProcessor


class FrozenStyleEncoder:
    # 非 nn.Module：共享冻结骨干不能注册到新分支优化器或 checkpoint。
    def __init__(self, image_encoder, arm):
        self.model = image_encoder
        self.vision = image_encoder.vision_model
        self.arm = arm
        self.training = False
        self.reset()

    def reset(self):
        self.rng = torch.Generator(device='cpu').manual_seed(49042)

    @torch.no_grad()
    def encode_pixels(self, pixels, permutation=None):
        v = self.vision
        if self.arm == 'CLIP':
            hidden = v.embeddings(pixels)
        else:
            e = v.embeddings
            patches = e.patch_embedding(pixels.to(e.patch_embedding.weight.dtype)).flatten(2).transpose(1, 2)
            assert permutation is not None and permutation.shape == patches.shape[:2]
            # 去掉 CLS，并在 Transformer 之前打乱 patch 位置编码；像素 patch 顺序保持。
            positions = e.position_embedding.weight[1:][permutation.to(patches.device)]
            hidden = patches + positions
        hidden = v.pre_layrnorm(hidden)
        return v.encoder(inputs_embeds=hidden, output_attentions=False,
            output_hidden_states=False, return_dict=True)[0].detach()

    @torch.no_grad()
    def __call__(self, pixels):
        assert pixels.shape[0] == 1
        rgb = ((pixels[0].float().cpu() + 1) / 2).clamp(0, 1)
        if self.training:
            rng = self.rng
        else:
            # 推理与调用次序无关；两臂 crop 完全相同，也不消耗扩散/训练全局 RNG。
            digest = hashlib.sha256(rgb.numpy().tobytes()).digest()
            rng = torch.Generator(device='cpu').manual_seed(int.from_bytes(digest[:8], 'little') % (2**63-1))
        _, h, w = rgb.shape
        ch, cw = round(h*.8), round(w*.8)
        crops = [rgb]
        boxes = [[0, 0, w, h]]
        for _ in range(2):
            top = int(torch.randint(h-ch+1, (), generator=rng))
            left = int(torch.randint(w-cw+1, (), generator=rng))
            crops.append(rgb[:, top:top+ch, left:left+cw]); boxes.append([left, top, left+cw, top+ch])
        # 使用 CLIP 标准归一化；两臂同样 resize，不用会改变条纹方向的翻转/旋转。
        side = int(self.model.config.image_size)
        batch = torch.cat([F.interpolate(c[None], size=(side, side), mode='bicubic',
            align_corners=False, antialias=True).clamp(0, 1) for c in crops])
        mean = batch.new_tensor([.48145466, .4578275, .40821073])[None, :, None, None]
        std = batch.new_tensor([.26862954, .26130258, .27577711])[None, :, None, None]
        batch = ((batch-mean)/std).to(device=pixels.device, dtype=self.vision.embeddings.patch_embedding.weight.dtype)
        n = self.vision.embeddings.position_embedding.weight.shape[0]-1
        # 普通臂同样抽取 permutation，确保训练 crop RNG 消费量一致。
        permutation = torch.stack([torch.randperm(n, generator=rng) for _ in crops])
        result = self.encode_pixels(batch, permutation)
        self.last_audit = dict(crop_boxes=boxes, position_permutation_sha256=hashlib.sha256(
            permutation.numpy().tobytes()).hexdigest(), token_shape=list(result.shape),
            cls_removed=self.arm=='SSVM', shuffled_before_encoder=self.arm=='SSVM')
        return result.reshape(1, -1, result.shape[-1])


class StyleReferenceProcessor(IndependentReferenceProcessor):
    def __init__(self, original, name, width, clip_width):
        super().__init__(original, name, width, width, 'B1')
        # 复用现有 Perceiver；小支路适配，不声称复现论文的 316M 模型。
        self.style_embed = PerceiverResampler(dim=width, depth=1, dim_head=64,
            heads=width//64, num_latents=16, embedding_dim=clip_width,
            output_dim=width, ff_mult=2)
