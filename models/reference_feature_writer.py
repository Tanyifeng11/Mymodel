"""冻结参考 U-Net 的逐 timestep 空间读取；BF-local 保留原条件路径。"""
import torch
from torch.nn import functional as F


def spatial_tokens(feature, spatial_shape):
    if feature.ndim == 3:
        b, _, c = feature.shape
        feature = feature.transpose(1, 2).reshape(b, c, *spatial_shape)
    return F.adaptive_avg_pool2d(feature, (16, 16)).flatten(2).transpose(1, 2)


class ReferenceFeatureWriter:
    def __init__(self, pipe, layer, model_path):
        from diffusers import UNet2DConditionModel
        self.pipe, self.layer = pipe, layer
        self.unet = UNet2DConditionModel.from_pretrained(model_path, subfolder='unet',
            torch_dtype=pipe.unet.dtype, local_files_only=True).to(pipe.device)
        self.unet.eval().requires_grad_(False)
        self.feature = None
        self.calls = 0
        module = dict(self.unet.named_modules())[layer[:-len('.processor')]]
        self.handle = module.register_forward_pre_hook(self.capture)
        self.latent = self.text = None

    def capture(self, module, args):
        self.feature = args[0].detach()

    @torch.no_grad()
    def set_reference(self, pixels, text):
        self.latent = self.pipe.vae.encode(pixels.to(self.pipe.vae.dtype)).latent_dist.mean
        self.latent = self.latent * self.pipe.vae.config.scaling_factor
        self.text = text

    @torch.no_grad()
    def write(self, timestep):
        self.feature = None
        self.unet(self.latent, timestep, encoder_hidden_states=self.text, return_dict=False)
        self.calls += 1
        assert self.feature is not None and torch.isfinite(self.feature).all()
        # 此层在当前 SD1.5 中处于 latent 的半分辨率，实际 shape 由 P0 记录验证。
        hw = (self.latent.shape[-2] // 2, self.latent.shape[-1] // 2)
        return spatial_tokens(self.feature, hw)

    def close(self):
        self.handle.remove()
