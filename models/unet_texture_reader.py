"""在一处 U-Net 注意力层读取二维参考纹理。"""
from contextlib import contextmanager

import torch
from torch import nn
import torch.nn.functional as F


ATTENTION = 'up_blocks.2.attentions.0.transformer_blocks.0.attn1'
QUERY_HW = (32, 24)


class TextureReader(nn.Module):
    def __init__(self, channels=640, width=128, heads=4):
        super().__init__()
        self.heads = heads
        self.patch = nn.Conv2d(4, width, kernel_size=5, padding=2)
        self.query = nn.Linear(channels, width, bias=False)
        self.keys_values = nn.Linear(width, 2 * width, bias=False)
        self.output = nn.Linear(width, channels, bias=False)
        nn.init.zeros_(self.output.weight)
        self.grid = self.present = self.cached = None
        self.calls = 0

    def keys_and_values(self, grid):
        # 先在 VAE 网格上读取局部邻域，再将各位置展成序列。
        patches = F.silu(self.patch(grid)).flatten(2).transpose(1, 2)
        return self.keys_values(patches).chunk(2, dim=-1)

    def forward(self, states):
        assert states.ndim == 3 and states.shape[1] == QUERY_HW[0] * QUERY_HW[1], states.shape
        self.calls += 1
        key, value = self.cached if self.cached is not None else self.keys_and_values(self.grid)

        def split_heads(tensor):
            return tensor.unflatten(-1, (self.heads, -1)).transpose(1, 2)

        result = F.scaled_dot_product_attention(
            split_heads(self.query(states)), split_heads(key), split_heads(value))
        residual = self.output(result.transpose(1, 2).flatten(2))
        return residual if self.present is None else residual * self.present

    @contextmanager
    def inference(self, grid, shuffle_seed=None):
        assert self.cached is None and self.grid is None
        if shuffle_seed is not None:
            generator = torch.Generator().manual_seed(shuffle_seed)
            order = torch.randperm(grid.shape[-2] * grid.shape[-1], generator=generator)
            grid = grid.flatten(2).index_select(2, order.to(grid.device)).reshape_as(grid)
        self.calls = 0
        self.cached = self.keys_and_values(grid)
        try:
            yield
        finally:
            self.cached = None


@torch.no_grad()
def encode_reference(vae, reference):
    # 用后验均值缓存参考，编码不消耗采样随机数。
    return vae.encode(reference.to(dtype=vae.dtype)).latent_dist.mean * vae.config.scaling_factor


def attach_reader(network, vae):
    attention = network.unet.get_submodule(ATTENTION)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(1112)
        reader = TextureReader(attention.to_q.in_features)
    reader.to(attention.to_q.weight)
    network.add_module('texture_reader', reader)

    def condition(_module, inputs):
        reference = inputs[4]
        reader.grid = encode_reference(vae, reference)
        # 保持 E2 的参考 dropout。零输入仍经过支路，保证 DDP 梯度同步。
        reader.present = reference.flatten(1).ne(0).any(dim=1).view(-1, 1, 1)
        reader.calls = 0

    def clear(_module, _inputs, _output):
        reader.grid = reader.present = None

    def add_residual(_module, inputs, kwargs, output):
        if network.adapter.conditional and (reader.grid is not None or reader.cached is not None):
            states = inputs[0] if inputs else kwargs['hidden_states']
            return output + reader(states).to(output.dtype)
        return output

    network.register_forward_pre_hook(condition)
    network.register_forward_hook(clear, always_call=True)
    attention.register_forward_hook(add_residual, with_kwargs=True)
    return reader

