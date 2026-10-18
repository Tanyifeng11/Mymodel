"""原M10完整统计残差的可学习强度；0步精确保持原M10。"""
import torch
from torch import nn


class M10Gain(nn.Module):
    def __init__(self, width=640):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(width))
        self.b = nn.Parameter(torch.zeros(width))

    def forward(self, timestep):
        time = torch.as_tensor(timestep, device=self.a.device, dtype=torch.float32).reshape(-1, 1, 1)
        return 2 * torch.sigmoid(self.a[None, None] + self.b[None, None] * (2 * time / 999 - 1))
