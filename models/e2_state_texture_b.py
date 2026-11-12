"""根据当前去噪状态修正 x0；零初始化保持原采样器。"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class StateTextureB(nn.Module):
    def __init__(self, width=64, color_channels=0):
        super().__init__()
        self.input = nn.Conv2d(4 + 4 + 64 + 1 + 1 + color_channels, width, 3, padding=1)
        self.time = nn.Sequential(nn.Linear(32, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList([nn.Sequential(nn.GroupNorm(8, width), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(8, width), nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1)) for _ in range(2)])
        self.output = nn.Conv2d(width, 4, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, zt, x0, reference, mask, structure, timestep, color=None):
        frequency = torch.exp(-math.log(10000) * torch.arange(16, device=zt.device) / 15)
        angle = timestep.float().reshape(-1, 1) * frequency[None]
        embedding = self.time(torch.cat((angle.sin(), angle.cos()), 1))[:, :, None, None]
        conditions = [zt, x0, reference, mask, structure]
        if color is not None:
            conditions.append(color)
        hidden = self.input(torch.cat(conditions, 1)) + embedding
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return self.output(F.silu(hidden)) * mask
