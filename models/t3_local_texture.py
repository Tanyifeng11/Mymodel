"""冻结 T3 的局部参考残差；零输出起步，只有新支路参与训练。"""
import torch
from torch import nn
import torch.nn.functional as F


class TextureResidual(nn.Module):
    def __init__(self, channels, width=64):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.query = nn.Linear(channels, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.time = nn.Sequential(nn.Linear(2, width), nn.SiLU(), nn.Linear(width, width))
        self.output = nn.Conv2d(width, channels, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, hidden, reference, mask, timestep):
        b, _, h, w = hidden.shape
        q = self.query(self.norm(hidden.float().flatten(2).transpose(1, 2)))
        t = timestep.float().reshape(-1, 1) / 999
        if len(t) == 1:
            t = t.expand(b, -1)
        q = q + self.time(torch.cat([t, t.square()], 1))[:, None]
        k, v = self.key(reference), self.value(reference)
        # 四头注意力保留参考局部特征；不把参考画布坐标当作衣身坐标。
        reshape = lambda x: x.reshape(b, -1, 4, q.shape[-1] // 4).transpose(1, 2)
        detail = F.scaled_dot_product_attention(reshape(q), reshape(k), reshape(v))
        detail = detail.transpose(1, 2).reshape(b, h, w, -1).permute(0, 3, 1, 2)
        delta = self.output(detail)
        # 输出端限制作用域；仍需终图结构评价，不能据此保证背景不变。
        small = F.interpolate(mask.float(), (h, w), mode='area')
        small = (small > .999).float()
        inner = -F.max_pool2d(-small, 3, 1, 1)
        feather = F.avg_pool2d(inner, 3, 1, 1) * inner
        return (delta * feather).to(hidden.dtype)


class LocalTextureBranch(nn.Module):
    layers = ('up_blocks.2.resnets.1', 'up_blocks.3.resnets.1')

    def __init__(self, channels=(640, 320), width=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, width, 3, 2, 1), nn.GroupNorm(8, width), nn.SiLU(),
            nn.Conv2d(width, width, 3, 2, 1), nn.GroupNorm(8, width), nn.SiLU())
        self.blocks = nn.ModuleList([TextureResidual(c, width) for c in channels])
        self.handles = []
        self.enabled = False
        self.conditional = False
        self.context = self.mask = self.timestep = None

    def condition(self, image, mask):
        self.context = F.adaptive_avg_pool2d(self.encoder(image.float()), (16, 16)).flatten(2).transpose(1, 2)
        self.mask = mask.float()

    def bind(self, unet):
        def before(module, args, kwargs):
            extra = kwargs.get('cross_attention_kwargs') or {}
            self.conditional = 'sa_hidden_states' in extra
            self.timestep = kwargs.get('timestep', args[1] if len(args) > 1 else None)
            if not torch.is_tensor(self.timestep):
                self.timestep = self.mask.new_tensor([self.timestep])
        self.handles.append(unet.register_forward_pre_hook(before, with_kwargs=True))
        for i, layer in enumerate(self.layers):
            def inject(module, args, output, index=i):
                if (not self.enabled or not self.conditional or self.context is None
                        or int(self.timestep.flatten()[0]) > 249):
                    return output
                ref, mask = self.context, self.mask
                if len(output) == 2 * len(ref):
                    ref = torch.cat([ref, ref])
                    mask = torch.cat([torch.zeros_like(mask), mask])
                return output + self.blocks[index](output, ref, mask, self.timestep)
            self.handles.append(unet.get_submodule(layer).register_forward_hook(inject))

    def clear(self):
        self.context = self.mask = self.timestep = None

    def unbind(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.clear()


def whiten(image):
    gray = (image.float() * image.new_tensor([.2126, .7152, .0722])[None, :, None, None]).sum(1, keepdim=True)
    return (gray - gray.mean((-2, -1), keepdim=True)) / gray.std((-2, -1), keepdim=True, unbiased=False).clamp_min(.02)


def power_descriptor(patches):
    x = whiten(patches)
    side = x.shape[-1]
    win = torch.hann_window(side, periodic=False, device=x.device)
    power = torch.fft.fft2(x * win[:, None] * win[None, :]).abs().square()
    power = torch.fft.fftshift(power, dim=(-2, -1))
    keep = torch.ones_like(power)
    keep[..., side // 2 - 1:side // 2 + 2, side // 2 - 1:side // 2 + 2] = 0
    power = F.avg_pool2d(power * keep, 3, 1, 1)
    power = power.flatten(1) + 1e-8
    return torch.sqrt(power / power.sum(1, keepdim=True))


def reference_consistency(predicted, reference):
    """平移不敏感的灰度频谱集合距离；不宣称这是纹样身份真值。"""
    a, b = power_descriptor(predicted), power_descriptor(reference)
    distances = (a[:, None] - b[None]).square().sum(-1)
    return .5 * (distances.min(1).values.mean() + distances.min(0).values.mean())
